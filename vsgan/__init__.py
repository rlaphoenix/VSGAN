from __future__ import annotations

import functools
from typing import Union, Optional

import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.archs import ESRGAN
from vsgan.constants import MAX_DTYPE_VALUES


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
        if not isinstance(clip, vs.VideoNode):
            raise ValueError("VSGAN: This is not a clip")

        if clip.format.color_family.name != "RGB":
            raise ValueError("VSGAN: Only RGB clips are supported. RGB24 or RGBS recommended.")

        device = device.strip().lower() if isinstance(device, str) else device
        if device == "":
            raise ValueError("VSGAN: Device must be provided.")
        if device == "cpu":
            raise UserWarning(
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
        self.half: bool = False

    def load_model(self, model: str, half: bool = False) -> VSGAN:
        """
        Load a model file and send to the PyTorch device. The model can be
        changed at any point.

        Args:
            model: Path to a supported PyTorch Model file.
            half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
        """
        model = ESRGAN(model)
        model.eval()
        self.model = model.to(self.device)
        if half:
            self.model.half()
        self.half = half
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

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model.scale,
                height=self.clip.height * self.model.scale
            ),
            functools.partial(
                self.execute,
                clip=self.clip,
                model=self.model,
                half=self.half,
                overlap=overlap,
            )
        )

        return self

    @torch.inference_mode()
    def execute(self, n: int, clip: vs.VideoNode, model: torch.nn.Module, half: bool = False,
                overlap: int = 0) -> vs.VideoNode:
        """
        Run the ESRGAN repo's Modified ESRGAN RRDBNet super-resolution code on a clip's frame.
        Unlike the original code, frames are modified directly as Tensors, without CV2.

        Thanks to VideoHelp for initial support, and @JoeyBallentine for his work on
        seamless chunk support.
        """
        lr_img = self.frame_to_tensor(clip.get_frame(n), change_order=True, add_batch=True, half=half)
        try:
            if not overlap:
                output_img = model(lr_img.to(self.device)).data
            elif overlap > 0:
                # seamless tiles
                b, c, h, w = lr_img.shape

                out_h = h * model.scale
                out_w = w * model.scale
                output_img = torch.empty(
                    (b, c, out_h, out_w), dtype=lr_img.dtype, device=lr_img.device
                )

                top_left_sr = model(lr_img[..., : h // 2 + overlap, : w // 2 + overlap].to(self.device)).data
                top_right_sr = model(lr_img[..., : h // 2 + overlap, w // 2 - overlap:].to(self.device)).data
                bottom_left_sr = model(lr_img[..., h // 2 - overlap:, : w // 2 + overlap].to(self.device)).data
                bottom_right_sr = model(lr_img[..., h // 2 - overlap:, w // 2 - overlap:].to(self.device)).data

                output_img[..., : out_h // 2, : out_w // 2] = top_left_sr[..., : out_h // 2, : out_w // 2]
                output_img[..., : out_h // 2, -out_w // 2:] = top_right_sr[..., : out_h // 2, -out_w // 2:]
                output_img[..., -out_h // 2:, : out_w // 2] = bottom_left_sr[..., -out_h // 2:, : out_w // 2]
                output_img[..., -out_h // 2:, -out_w // 2:] = bottom_right_sr[..., -out_h // 2:, -out_w // 2:]
            else:
                raise ValueError("Invalid overlap. Must be a value greater than 0, or a False-y value to disable.")
        except RuntimeError as e:
            if "allocate" in str(e) or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
            raise

        return self.tensor_to_clip(clip, output_img)

    @staticmethod
    def frame_to_np(frame: vs.VideoFrame) -> np.dstack:
        """
        Alternative to cv2.imread() that will directly read images to a numpy array.
        :param frame: VapourSynth frame from a clip
        """
        return np.dstack([np.asarray(frame[i]) for i in range(frame.format.num_planes)])

    @staticmethod
    def frame_to_tensor(frame: vs.VideoFrame, change_range=True, change_order=False, bgr2rgb=False, add_batch=False,
                        normalize=False, half: bool = False) -> torch.Tensor:
        """
        Read an image as a numpy array and convert it to a tensor.
        :param frame: VapourSynth frame from a clip.
        :param normalize: Normalize (z-norm) from [0,1] range to [-1,1].
        :param half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
        """
        array = VSGAN.frame_to_np(frame)

        if change_range:
            max_val = MAX_DTYPE_VALUES.get(array.dtype, 1.0)
            array = array.astype(np.dtype("float32")) / max_val

        if change_order:
            array = np.transpose(array, (2, 0, 1))  # HWC->CHW

        array = torch.from_numpy(np.ascontiguousarray(array)).float()

        if half:
            array = array.half()

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
        array = t.squeeze(0).detach().clamp(0, 1).cpu().numpy()

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
        clip = core.std.BlankClip(
            clip=clip,
            width=image.shape[-1],
            height=image.shape[-2]
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: self.tensor_to_frame(f.copy(), image)
        )
