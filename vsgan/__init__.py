import functools

import mvsfunc
import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core


class VSGAN:

    def __init__(self, device="cuda"):
        self.device = device.lower() if isinstance(device, str) else device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is not available, reverting to \"cpu\" as device...")
            self.device = "cpu"
        self.torch_device = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model):
        state_dict = torch.load(model)

        # Check if new-arch and convert
        if "conv_first.weight" in state_dict:
            state_dict = self.convert_new_to_old(state_dict)

        # extract model information
        scale2 = 0
        max_part = 0
        scalemin = 6
        for part in list(state_dict):
            parts = part.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > scalemin and parts[0] == "model" and parts[2] == "weight":
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        upscale = 2 ** scale2
        in_nc = state_dict["model.0.weight"].shape[1]
        nf = state_dict["model.0.weight"].shape[0]
        self.model_scale = upscale

        self.rrdb_net_model = self.get_rrdb_net_arch(in_nc, out_nc, nf, nb)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()

        if not self.torch_device:
            # only need to load a torch device once, and only when loading a model
            self.torch_device = torch.device(self.device)
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip, chunk=False):
        # convert clip to RGB24 as it cannot read any other color space
        buffer = mvsfunc.ToRGB(clip, depth=8, kernel="spline36")  # expecting RGB24 (RGB 8bpp)
        # send the clip array to execute()
        results = []
        for c in self.chunk_clip(buffer) if chunk else [buffer]:
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
        buffer = core.std.StackHorizontal([
            core.std.StackVertical([results[0], results[1]]),
            core.std.StackVertical([results[2], results[3]])
        ]) if chunk else results[0]

        # VSGAN used to convert back to the original color space which resulted
        # in a LOT of guessing, which was in-accurate and may not be efficient
        # depending on what the user is doing after running VSGAN, so in all
        # versions after 1.0.6-post1 we return it in the color-space the GAN
        # provides which is always RGB24.

        # return the new frame
        return buffer
    
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

    @staticmethod
    def chunk_clip(clip):
        # split the clip horizontally into 2 images
        crops = {
            "left": core.std.CropRel(clip, left=0, top=0, right=clip.width / 2, bottom=0),
            "right": core.std.CropRel(clip, left=clip.width / 2, top=0, right=0, bottom=0)
        }
        # split each of the 2 images from above, vertically, into a further 2 images (totalling 4 images per frame)
        # top left, bottom left, top right, bottom right
        return [
            core.std.CropRel(crops["left"], left=0, top=0, right=0, bottom=crops["left"].height / 2),
            core.std.CropRel(crops["left"], left=0, top=crops["left"].height / 2, right=0, bottom=0),
            core.std.CropRel(crops["right"], left=0, top=0, right=0, bottom=crops["right"].height / 2),
            core.std.CropRel(crops["right"], left=0, top=crops["right"].height / 2, right=0, bottom=0)
        ]

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
