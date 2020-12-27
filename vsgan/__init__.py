import functools
import itertools

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core


class VSGAN:

    def __init__(self, device="cuda"):
        device = device.lower() if isinstance(device, str) else device
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
            raise EnvironmentError(
                "VSGAN: CUDA is not available, make sure you installed NVIDIA CUDA and that your GPU is available.")
        self.device = device
        self.torch_device = torch.device(self.device)
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model):
        state_dict = torch.load(model)
        if "conv_first.weight" in state_dict:
            # model is "new-arch", convert to old state dict structure
            state_dict = self.convert_new_to_old(state_dict)
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

        self.rrdb_net_model = self.get_rrdb_net_arch(in_nc, out_nc or in_nc, nf, nb)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip, chunk=False):
        if clip.format.color_family.name != "RGB":
            raise ValueError(
                "VSGAN: Clip color format must be RGB as the ESRGAN model can only work with RGB data :(\n"
                "You can use mvsfunc.ToRGB or use the format option on core.resize functions.\n"
                "The clip might need to be bit depth of 8bpp for correct color input/output.\n"
                "If you need to specify a kernel for chroma, I recommend Spline or Bicubic."
            )
        # send the clip array to execute()
        results = []
        for c in self.chunk_clip(clip) if chunk else [clip]:
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

    @staticmethod
    def convert_new_to_old(state_dict):
        old_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        old_net["model.0.weight"] = state_dict["conv_first.weight"]
        old_net["model.0.bias"] = state_dict["conv_first.bias"]

        for k in items.copy():
            if "RDB" in k:
                ori_k = k.replace("RRDB_trunk.", "model.1.sub.")
                if ".weight" in k:
                    ori_k = ori_k.replace(".weight", ".0.weight")
                elif ".bias" in k:
                    ori_k = ori_k.replace(".bias", ".0.bias")
                old_net[ori_k] = state_dict[k]
                items.remove(k)

        old_net["model.1.sub.23.weight"] = state_dict["trunk_conv.weight"]
        old_net["model.1.sub.23.bias"] = state_dict["trunk_conv.bias"]
        old_net["model.3.weight"] = state_dict["upconv1.weight"]
        old_net["model.3.bias"] = state_dict["upconv1.bias"]
        old_net["model.6.weight"] = state_dict["upconv2.weight"]
        old_net["model.6.bias"] = state_dict["upconv2.bias"]
        old_net["model.8.weight"] = state_dict["HRconv.weight"]
        old_net["model.8.bias"] = state_dict["HRconv.bias"]
        old_net["model.10.weight"] = state_dict["conv_last.weight"]
        old_net["model.10.bias"] = state_dict["conv_last.bias"]
        return old_net

    def get_rrdb_net_arch(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        """
        Import RRDB Net Architecture

        :param in_nc: num of input channels
        :param out_nc: num of output channels
        :param nf: num of filters
        :param nb: num of blocks
        :param gc: ?
        """

        from . import RRDBNet_arch_old as Arch
        return Arch.RRDB_Net(
            in_nc, out_nc, nf, nb, gc,
            upscale=self.model_scale,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv"
        )

    def chunk_clip(self, clip):
        """
        Split clip down the center into two clips (a left and right clip)
        Then split those 2 clips in the center into two clips (a top and bottom clip).
        Resulting in a total of 4 clips (aka chunk).
        """
        return itertools.chain.from_iterable([
            self.split(x, axis=1) for x in self.split(clip, axis=0)
        ])

    @staticmethod
    def split(clip, axis):
        if axis == 0:
            return [
                core.std.CropAbs(clip, left=0, right=clip.width / 2),
                core.std.CropAbs(clip, left=clip.width / 2, right=0)
            ]
        elif axis == 1:
            return [
                core.std.CropAbs(clip, top=0, bottom=clip.height / 2),
                core.std.CropAbs(clip, top=clip.height / 2, bottom=0)
            ]
        raise ValueError("Invalid split axis...")

    @staticmethod
    def cv2_imread(frame, plane_count):
        """
        Alternative to cv2.imread() that will directly read images to a numpy array
        """
        return np.dstack(
            [np.array(frame.get_read_array(i), copy=False) for i in reversed(range(plane_count))]
        )

    @staticmethod
    def cv2_imwrite(image, out_color_space="RGB24"):
        """
        Alternative to cv2.imwrite() that will convert the data into an image readable by VapourSynth
        """
        if len(image.shape) <= 3:
            image = image.reshape([1] + list(image.shape))
        # Define the shapes items
        plane_count = image.shape[-1]
        image_width = image.shape[-2]
        image_height = image.shape[-3]
        image_length = image.shape[-4]
        # this is a clip (or array buffer for frames) that we will insert the GAN'd frames into
        buffer = core.std.BlankClip(
            clip=None,
            width=image_width,
            height=image_height,
            format=vs.PresetFormat[out_color_space],
            length=image_length
        )

        def replace_planes(n, f):
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

    def execute(self, n, clip):
        """
        Essentially the same as ESRGAN, except it replaces the cv2 functions with ones geared towards VapourSynth
        https://github.com/xinntao/ESRGAN/blob/master/test.py#L26
        """
        if not self.rrdb_net_model or not self.torch_device:
            raise Exception("Error: Model not yet loaded a torch device...")
        # get the frame being used
        frame = clip.get_frame(n)
        img = self.cv2_imread(frame=frame, plane_count=clip.format.num_planes)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.torch_device)
        with torch.no_grad():
            output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        return self.cv2_imwrite(image=output, out_color_space=clip.format.name)
