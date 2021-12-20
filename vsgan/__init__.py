from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Union, Optional

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

# noinspection PyPep8Naming
from vsgan.models.ESRGAN import Network as ESRGAN
from vsgan.constants import MAX_DTYPE_VALUES

model_state_T = OrderedDict[str, torch.Tensor]


class VSGAN:
    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        """
        Create a PyTorch Device instance to use VSGAN with.
        It validates the supplied pytorch device identifier, and makes
        sure CUDA environment is available and ready.

        Args:
            clip: VapourSynth Video Node (aka clip). It must be RGB
                colorspace. RGB27 and RGB30 may not work.
            device: PyTorch device identifier to use for the model. E.g.,
                "cuda", "cpu", 0, 1, and so on.
        """
        device = device.strip().lower() if isinstance(device, str) else device
        if device == "":
            raise ValueError("VSGAN: `device` parameter cannot be an empty string.")
        if device == "cpu":
            raise ValueError(
                "VSGAN blocked an attempt to use your CPU as the torch device. "
                "Using your CPU will run it at very high utilisation and may lower its lifespan. "
                "If you are sure you would like to use your CPU, then use `cpu!`."
            )
        if device == "cpu!":
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise EnvironmentError("VSGAN: Either NVIDIA CUDA or the device (%s) isn't available." % device)

        self.clip: vs.VideoNode = clip
        self.device: torch.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.model_scale: Optional[int] = None

    def load_model(self, model: str) -> VSGAN:
        """
        Load a model file and send to the PyTorch device. The model can be
        changed at any point.

        Args:
            model: Path to a supported PyTorch Model file.
        """
        model_state = self.sanitize_state_dict(torch.load(model))

        scale2 = 0
        max_part = 0
        scale_min = 6
        nb = None
        out_nc = None
        for part in list(model_state):
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
                    out_nc = model_state[part].shape[0]

        self.model_scale = 2 ** scale2
        in_nc = model_state["model.0.weight"].shape[1]
        nf = model_state["model.0.weight"].shape[0]

        if nb is None:
            raise ValueError("Could not find the nb in this new-arch model.")
        if out_nc is None:
            print("[!] Could not find out_nc, assuming it's the same as in_nc...")

        model = ESRGAN(in_nc, out_nc or in_nc, nf, nb, self.model_scale)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        self.model = model.to(self.device)

        return self

    def run(self, overlap: int = 0) -> VSGAN:
        """
        Executes the model on each frame. It uses FrameEval as to not apply to
        every frame immediately, instead only upon request.

        The overlap value enables chunk-mode and specifies the amount to extend
        each quadrant as to hide seams. This effectively cuts VRAM requirements
        by 75%. Use this if you do not have enough VRAM for your input image.

        It should generally be a multiple of 16. The larger the input
        resolution, the larger overlap may need to be set. Avoid using a value
        excessively large.

        The larger the overlap value, the more VRAM you will use per-quadrant,
        and the slower it may perform.
        """
        if not self.model:
            raise ValueError("A model must be loaded before running.")

        if self.clip.format.color_family.name != "RGB":
            raise ValueError(
                "VSGAN only supports RGB clips. RGB24 or RGBS recommended. "
                "You can use core.resize.* funcs to convert."
            )

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model_scale,
                height=self.clip.height * self.model_scale
            ),
            functools.partial(
                self.execute,
                clip=self.clip,
                model=self.model,
                scale=self.model_scale,
                overlap=overlap
            )
        )

        return self

    def execute(self, n: int, clip: vs.VideoNode, model: torch.nn.Module, scale: int, overlap: int = 0) -> vs.VideoNode:
        """
        Run the ESRGAN repo's Modified ESRGAN RRDBNet super-resolution code on a clip's frame.
        Unlike the original code, frames are modified directly as Tensors, without CV2.

        Thanks to VideoHelp for initial support, and @JoeyBallentine for his work on
        seamless chunk support.
        """

        def run_model(quadrant: torch.Tensor) -> torch.Tensor:
            try:
                quadrant = quadrant.to(self.device)
                with torch.no_grad():
                    return model(quadrant).data
            except RuntimeError as e:
                if "allocate" in str(e) or "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise

        lr_img = self.frame_to_tensor(clip.get_frame(n))

        if not overlap:
            output_img = run_model(lr_img)
        elif overlap > 0:
            b, c, h, w = lr_img.shape

            out_h = h * scale
            out_w = w * scale
            output_img = torch.empty(
                (b, c, out_h, out_w), dtype=lr_img.dtype, device=lr_img.device
            )

            top_left_sr = run_model(lr_img[..., : h // 2 + overlap, : w // 2 + overlap])
            top_right_sr = run_model(lr_img[..., : h // 2 + overlap, w // 2 - overlap:])
            bottom_left_sr = run_model(lr_img[..., h // 2 - overlap:, : w // 2 + overlap])
            bottom_right_sr = run_model(lr_img[..., h // 2 - overlap:, w // 2 - overlap:])

            output_img[..., : out_h // 2, : out_w // 2] = top_left_sr[..., : out_h // 2, : out_w // 2]
            output_img[..., : out_h // 2, -out_w // 2:] = top_right_sr[..., : out_h // 2, -out_w // 2:]
            output_img[..., -out_h // 2:, : out_w // 2] = bottom_left_sr[..., -out_h // 2:, : out_w // 2]
            output_img[..., -out_h // 2:, -out_w // 2:] = bottom_right_sr[..., -out_h // 2:, -out_w // 2:]
        else:
            raise ValueError("Invalid overlap. Must be a value greater than 0, or a False-y value to disable.")

        return self.tensor_to_clip(clip, output_img)

    @staticmethod
    def sanitize_state_dict(state_dict: model_state_T) -> model_state_T:
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
        old_net = OrderedDict({
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
        })
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
    def frame_to_tensor(frame: vs.VideoFrame, change_range=True, bgr2rgb=False, add_batch=True, normalize=False) \
            -> torch.Tensor:
        """
        Read an image as a numpy array and convert it to a tensor.
        :param frame: VapourSynth frame from a clip.
        :param normalize: Normalize (z-norm) from [0,1] range to [-1,1].
        """
        array = VSGAN.frame_to_np(frame)

        if change_range:
            max_val = MAX_DTYPE_VALUES.get(array.dtype, 1.0)
            array = array.astype(np.dtype("float32")) / max_val

        array = torch.from_numpy(
            np.ascontiguousarray(np.transpose(array, (2, 0, 1)))  # HWC->CHW
        ).float()

        if bgr2rgb:
            if array.shape[0] % 3 == 0:
                # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
                array = array.flip(-3)
            elif array.shape[0] == 4:
                # RGBA
                array = array[[2, 1, 0, 3], :, :]

        if add_batch:
            # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
            array.unsqueeze_(0)

        if normalize:
            array = ((array - 0.5) * 2.0).clamp(-1, 1)

        return array

    @staticmethod
    def tensor_to_frame(f: vs.VideoFrame, t: torch.Tensor) -> vs.VideoFrame:
        """
        Copies each channel from a Tensor into a vs.VideoFrame.
        It expects the tensor array to have the dimension count (C) first in the shape, so CHW or CWH.
        :param f: VapourSynth frame to store retrieved planes.
        :param t: Tensor array to retrieve planes from.
        :returns: New frame with planes from tensor array
        """
        array = t.squeeze(0).detach().cpu().clamp(0, 1).numpy()

        d_type = np.asarray(f[0]).dtype
        array = MAX_DTYPE_VALUES.get(d_type, 1.0) * array
        array = array.astype(d_type)

        for plane in range(f.format.num_planes):
            d = np.asarray(f[plane])
            np.copyto(d, array[plane, :, :])
        return f

    def tensor_to_clip(self, clip: vs.VideoNode, image: torch.Tensor) -> vs.VideoNode:
        """
        Convert a tensor into a VapourSynth clip.
        :param clip: used to inherit expected return properties only
        :param image: tensor (expecting CHW shape order)
        :returns: VapourSynth clip with the frame applied
        """
        batch, planes, height, width = image.size()
        clip = core.std.BlankClip(
            clip=clip,
            width=width,
            height=height
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: self.tensor_to_frame(f.copy(), image)
        )
