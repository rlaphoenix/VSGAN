import functools
import itertools
from typing import Union, Iterable, List

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.RRDBNet_arch_old import RRDB_Net


class VSGAN:

    def __init__(self, device: Union[str, int] = "cuda"):
        """
        Create a PyTorch Device instance, to use VSGAN with. It validates the supplied pytorch device identifier
        for you, and makes sure CUDA environment is available and ready.
        :param device: PyTorch device identifier, tells VSGAN which device to run ESRGAN with. e.g. `cuda`, `0`, `1`
        """
        device = device.strip().lower() if isinstance(device, str) else device
        if device == "":
            raise ValueError("VSGAN: `device` parameter cannot be an empty string.")
        if device == "cpu":
            raise ValueError(
                "VSGAN: Using your CPU as a device for VSGAN/PyTorch has been blocked, use a GPU device.\n"
                "Using ESRGAN on a CPU will run it at very high utilisation and temps and may straight up kill it.\n"
                "It isn't worth it either as it takes literally hours for a single 720x480 frame.\n"
                "If you are sure you would like to use your CPU, then use `cpu!` as the device argument."
            )
        elif device == "cpu!":
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise EnvironmentError(f"VSGAN: Either NVIDIA CUDA or the device ({device}) isn't available.")
        self.device = device
        self.torch_device = torch.device(self.device)
        self.model = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model: str):
        """
        Load an ESRGAN model file into the VSGAN object instance. The model can be changed by calling load_model
        at any point.
        :param model: ESRGAN .pth model file.
        """
        self.model = model
        state_dict = self.sanitize_state_dict(torch.load(self.model))
        # extract model information
        scale2 = 0
        max_part = 0
        scale_min = 6
        nb = None
        out_nc = None
        for part in list(state_dict):
            parts = part.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > scale_min and parts[0] == "model" and parts[2] == "weight":
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        self.model_scale = 2 ** scale2
        in_nc = state_dict["model.0.weight"].shape[1]
        nf = state_dict["model.0.weight"].shape[0]

        if nb is None:
            raise NotImplementedError("VSGAN: Could not find the nb in this new-arch model.")
        if out_nc is None:
            print("VSGAN Warning: Could not find out_nc, assuming it's the same as in_nc...")

        self.rrdb_net_model = self.get_rrdb_network(in_nc, out_nc or in_nc, nf, nb)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip: vs.VideoNode, chunk: bool = False) -> vs.VideoNode:
        """
        Executes VSGAN on the provided clip, returning the resulting in a new clip.
        :param clip: Clip to use as the input frames. It must be RGB. It will also return as RGB.
        :param chunk: Reduces VRAM usage by splitting the input frames into smaller sub-frames and renders them
        one by one, then merges them back together. Trading memory requirements for speed and accuracy.
        WARNING: The result may have issues on the edges of the chunks, example: https://imgbox.com/g/Hht5NqKB0i
        :returns: ESRGAN result clip
        """
        if clip.format.color_family.name != "RGB":
            raise ValueError(
                "VSGAN: Clip color format must be RGB as the ESRGAN model can only work with RGB data :(\n"
                "You can use mvsfunc.ToRGB or use the format option on core.resize functions.\n"
                "The clip might need to be bit depth of 8bpp for correct color input/output.\n"
                "If you need to specify a kernel for chroma, I recommend Spline or Bicubic."
            )
        # send the clip array to execute()
        results = []
        for c in self.chunk(clip) if chunk else [clip]:
            results.append(core.std.FrameEval(
                core.std.BlankClip(
                    clip=c,
                    width=c.width * self.model_scale,
                    height=c.height * self.model_scale
                ),
                functools.partial(
                    self.execute,
                    clip=c
                )
            ))
        # if chunked, rejoin the chunked clips otherwise return the result
        clip = core.std.StackHorizontal([
            core.std.StackVertical([results[0], results[1]]),
            core.std.StackVertical([results[2], results[3]])
        ]) if chunk else results[0]

        # return the new result clip
        return clip

    def execute(self, n: int, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Copies the xinntao ESRGAN repo's main execution code. The only real difference is it doesn't use cv2, and
        instead uses vapoursynth ports of cv2's functionality for read and writing "images".

        Code adapted from:
        https://github.com/xinntao/ESRGAN/blob/master/test.py#L26
        """
        if not self.rrdb_net_model:
            raise ValueError("VSGAN: No ESRGAN model has been loaded, use VSGAN.load_model().")
        # 255 being the max value for an RGB color space, could this be key to YUV support in the future?
        max_n = 255.0
        # what's up with all the different transposing of channel plane order?
        bgr = [2, 1, 0]
        brg = (2, 0, 1)
        gbr = (1, 2, 0)
        img = self.cv2_imread(frame=clip.get_frame(n), plane_count=clip.format.num_planes)
        img = img * 1.0 / max_n
        img = torch.from_numpy(np.transpose(img[:, :, bgr], brg)).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.torch_device)
        with torch.no_grad():
            output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[bgr, :, :], gbr)
        output = (output * max_n).round()
        return self.cv2_imwrite(image=output, out_color_space=clip.format.name)

    @staticmethod
    def sanitize_state_dict(state_dict: dict) -> dict:
        """
        Convert a new-arch model state dictionary to an old-arch dictionary.
        The new-arch model's only purpose is making the dict keys more verbose, but has no purpose other
        than that. So to easily support both new and old arch models, simply convert the key names back
        to their "Old" counterparts.

        :param state_dict: new-arch state dictionary
        :returns: old-arch state dictionary
        """
        if "conv_first.weight" not in state_dict:
            # model is already old arch, this is a loose check, but should be sufficient
            return state_dict
        old_net = {
            "model.0.weight": state_dict["conv_first.weight"],
            "model.0.bias": state_dict["conv_first.bias"],
            "model.1.sub.23.weight": state_dict["trunk_conv.weight"],
            "model.1.sub.23.bias": state_dict["trunk_conv.bias"],
            "model.3.weight": state_dict["upconv1.weight"],
            "model.3.bias": state_dict["upconv1.bias"],
            "model.6.weight": state_dict["upconv2.weight"],
            "model.6.bias": state_dict["upconv2.bias"],
            "model.8.weight": state_dict["HRconv.weight"],
            "model.8.bias": state_dict["HRconv.bias"],
            "model.10.weight": state_dict["conv_last.weight"],
            "model.10.bias": state_dict["conv_last.bias"]
        }
        for key, value in state_dict.items():
            if "RDB" in key:
                new = key.replace("RRDB_trunk.", "model.1.sub.")
                if ".weight" in key:
                    new = new.replace(".weight", ".0.weight")
                elif ".bias" in key:
                    new = new.replace(".bias", ".0.bias")
                old_net[new] = value
        return old_net

    def get_rrdb_network(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 23, gc: int = 32) -> RRDB_Net:
        """
        Create an old-arch style RRDB Network.
        :param in_nc: Number of input channels
        :param out_nc: Number of output channels
        :param nf: Number of filters
        :param nb: Number of blocks
        :param gc: ?
        """
        return RRDB_Net(
            in_nc, out_nc, nf, nb, gc,
            upscale=self.model_scale,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv"
        )

    def chunk(self, clip: vs.VideoNode) -> Iterable[vs.VideoNode]:
        """
        Split clip down the center into two clips (a left and right clip)
        Then split those 2 clips in the center into two clips (a top and bottom clip).
        Resulting in a total of 4 clips (aka chunk).
        """
        return itertools.chain.from_iterable([
            self.split(x, axis=1) for x in self.split(clip, axis=0)
        ])

    @staticmethod
    def split(clip: vs.VideoNode, axis: int) -> List[vs.VideoNode]:
        """
        Split a clip in the center of an axis, into two clips.
        :param clip: Clip to split
        :param axis: Axis to split on (0 = vertical, 1 = horizontal)
        """
        if axis == 0:
            return [
                core.std.Crop(clip, left=0, right=clip.width / 2),
                core.std.Crop(clip, left=clip.width / 2, right=0)
            ]
        elif axis == 1:
            return [
                core.std.Crop(clip, top=0, bottom=clip.height / 2),
                core.std.Crop(clip, top=clip.height / 2, bottom=0)
            ]
        raise ValueError("Invalid split axis...")

    @staticmethod
    def cv2_imread(frame: vs.VideoFrame, plane_count: int):
        """
        Alternative to cv2.imread() that will directly read images to a numpy array
        :param frame: VapourSynth frame from a clip
        :param plane_count: Amount of plane channels
        """
        return np.dstack(
            [np.array(frame.get_read_array(i), copy=False) for i in reversed(range(plane_count))]
        )

    @staticmethod
    def cv2_imwrite(image, out_color_space: str = "RGB24") -> vs.VideoNode:
        """
        Alternative to cv2.imwrite() that will convert the data into an image readable by VapourSynth
        :param image: Image data to save
        :param out_color_space: Color space to save the image in
        :returns: VapourSynth clip with the frame
        """
        if len(image.shape) <= 3:
            image = image.reshape([1] + list(image.shape))
        # Define the shapes items
        plane_count = image.shape[-1]
        # this is a clip (or array buffer for frames) that we will insert the GAN frames into
        buffer = core.std.BlankClip(
            width=image.shape[-2],
            height=image.shape[-3],
            format=vs.PresetFormat[out_color_space],
            length=image.shape[-4]
        )

        def replace_planes(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            """
            :param n: frame number
            :param f: frame
            :returns: frame with planes replaced
            """
            frame = f.copy()
            for i, plane_num in enumerate(reversed(range(plane_count))):
                # todo ; any better way to do this without storing the np.array in a variable?
                # todo ; perhaps some way to directly copy it to s?
                d = np.array(frame.get_write_array(plane_num), copy=False)
                # copy the value of d, into s[frame_num, :, :, plane_num]
                np.copyto(d, image[n, :, :, i], casting="unsafe")
                # delete the d variable from memory
                del d
            return frame

        # take the blank clip and insert the new data into the planes and return it back to sender
        return core.std.ModifyFrame(clip=buffer, clips=buffer, selector=replace_planes)
