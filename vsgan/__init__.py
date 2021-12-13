from __future__ import annotations

import functools
from typing import Union

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.RRDBNet import RRDBNet


class VSGAN:

    def __init__(self, device: Union[str, int] = "cuda"):
        """
        Create a PyTorch Device instance, to use VSGAN with.
        It validates the supplied pytorch device identifier, and makes sure CUDA environment is available and ready.
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
        if device == "cpu!":
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise EnvironmentError("VSGAN: Either NVIDIA CUDA or the device (%s) isn't available." % device)
        self.device = device
        self.torch_device = torch.device(self.device)
        self.model = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model: str):
        """
        Load an ESRGAN model file into the VSGAN object instance.
        The model can be changed by calling load_model at any point.
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

        self.rrdb_net_model = RRDBNet(in_nc, out_nc or in_nc, nf, nb, self.model_scale)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip: vs.VideoNode, overlap: int = 0) -> vs.VideoNode:
        """
        Executes VSGAN on the provided clip, returning the resulting in a new clip.
        :param clip: Clip to use as the input frames. It must be RGB. It will also return as RGB.
        :param overlap: Reduces VRAM usage by seamlessly rendering the input frame(s) in quadrants.
            This reduces memory usage but may also reduce speed. Only use this to stretch your VRAM.
        :returns: ESRGAN result clip
        """
        if clip.format.color_family.name != "RGB":
            raise ValueError(
                "VSGAN: Clip color format must be RGB as the ESRGAN model can only work with RGB data :(\n"
                "You can use mvsfunc.ToRGB or use the format option on core.resize functions.\n"
                "The clip might need to be bit depth of 8bpp for correct color input/output.\n"
                "If you need to specify a kernel for chroma, I recommend Spline or Bicubic."
            )

        return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip,
                width=clip.width * self.model_scale,
                height=clip.height * self.model_scale
            ),
            functools.partial(
                self.execute,
                clip=clip,
                overlap=overlap
            )
        )

    def execute(self, n: int, clip: vs.VideoNode, overlap: int = 0) -> vs.VideoNode:
        """
        Run the ESRGAN repo's main super-resolution code on a clip's frame.
        This function only contains about 10 lines of code from the original repo.
        Ported code via Numpy has been used inplace of cv2 for converting between arrays,
        tensors and vapoursynth VideoFrames and VideoNodes (clips).

        Thanks to VideoHelp for initial support, and @JoeyBallentine for his work on
        seamless chunk support.
        """
        if not self.rrdb_net_model:
            raise ValueError("VSGAN: No ESRGAN model has been loaded, use VSGAN.load_model().")

        def scale(quadrant: np.ndarray) -> np.ndarray:
            try:
                # original ESRGAN SR code by xinntao
                # https://github.com/xinntao/ESRGAN/blob/master/test.py#L26
                max_n = 255.0
                img = quadrant
                img = img * 1.0 / max_n
                img = np.transpose(img[:, :, (0, 1, 2)], (2, 0, 1))  # RGB to BRG
                img = torch.from_numpy(img).float()
                img_lr = img.unsqueeze(0).to(self.torch_device)
                with torch.no_grad():
                    output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[(2, 1, 0), :, :], (1, 2, 0))  # BGR to GBR
                output = (output * max_n).round()
                return output
                # end
            except RuntimeError as e:
                if "allocate" in str(e) or "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise

        lr_img = self.frame_to_np(clip.get_frame(n))

        if not overlap:
            output_img = scale(lr_img)
        elif overlap > 0:
            h, w, c = lr_img.shape

            out_h = h * self.model_scale
            out_w = w * self.model_scale
            output_img = np.zeros((out_h, out_w, c), np.uint8)

            top_left_sr = scale(lr_img[: h // 2 + overlap, : w // 2 + overlap, :])
            top_right_sr = scale(lr_img[: h // 2 + overlap, w // 2 - overlap:, :])
            bottom_left_sr = scale(lr_img[h // 2 - overlap:, : w // 2 + overlap, :])
            bottom_right_sr = scale(lr_img[h // 2 - overlap:, w // 2 - overlap:, :])

            output_img[: out_h // 2, : out_w // 2, :] = top_left_sr[: out_h // 2, : out_w // 2, :]
            output_img[: out_h // 2, -out_w // 2:, :] = top_right_sr[: out_h // 2, -out_w // 2:, :]
            output_img[-out_h // 2:, : out_w // 2, :] = bottom_left_sr[-out_h // 2:, : out_w // 2, :]
            output_img[-out_h // 2:, -out_w // 2:, :] = bottom_right_sr[-out_h // 2:, -out_w // 2:, :]
        else:
            raise ValueError("Invalid overlap. Must be a value greater than 0, or a False-y value to disable.")

        return self.np_to_clip(clip, output_img)

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

    @staticmethod
    def frame_to_np(frame: vs.VideoFrame) -> np.dstack:
        """
        Alternative to cv2.imread() that will directly read images to a numpy array.
        :param frame: VapourSynth frame from a clip
        """
        return np.dstack([np.asarray(frame[i]) for i in range(frame.format.num_planes)])

    @staticmethod
    def np_to_frame(f: vs.VideoFrame, array: np.ndarray, order: tuple = (2, 1, 0)) -> vs.VideoFrame:
        """
        Copies each channel from a numpy array into a vs.VideoFrame.
        It expects the numpy array to be BGR, with the dimension count (C) last in the shape, so HWC or WHC.
        :param f: VapourSynth frame to store retrieved planes.
        :param array: Numpy array to retrieve planes from.
        :param order: Specify input order of the numpy array color dimensions. It is most likely 2,1,0 (BGR).
        :returns: New frame with planes from numpy array
        """
        if list(order) != [0, 1, 2] and array.shape[-1] == 3:
            array = np.transpose(array[:, :, order], (0, 1, 2))  # `order` to RGB

        frame = f.copy()
        for plane in range(array.shape[-1]):
            d = np.array(frame[plane], copy=False)
            # TODO: Figure out why non-chunked execute() calls need unsafe casting
            np.copyto(d, array[:, :, plane], casting="unsafe")

        return frame

    def np_to_clip(self, clip: vs.VideoNode, image: np.ndarray, order: tuple = (2, 1, 0)) -> vs.VideoNode:
        """
        Convert a numpy array into a VapourSynth clip.
        :param clip: used to inherit expected return properties only
        :param image: numpy array (expecting HWC shape order)
        :param order: Specify input order of the numpy array color dimensions. It is most likely 2,1,0 (BGR).
        :returns: VapourSynth clip with the frame applied
        """
        height, width, _ = image.shape
        clip = core.std.BlankClip(
            clip=clip,
            width=width,
            height=height
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: self.np_to_frame(f, image, order)
        )
