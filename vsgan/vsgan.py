from __future__ import annotations

import functools
from typing import Union, Optional

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.archs import RealESRGANv2, ESRGAN
from vsgan.utilities import frame_to_tensor, tensor_to_clip, tile_tensor, join_tiles


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
        self.tensor_cache: dict = {}

    def load_model(self, model: str, half: bool = False) -> VSGAN:
        """
        Load a model file and send to the PyTorch device. The model can be
        changed at any point.

        Args:
            model: Path to a supported PyTorch Model file.
            half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
        """
        state = torch.load(model)
        if "params" in state and "body.0.weight" in state["params"]:
            arch = RealESRGANv2
        else:
            arch = ESRGAN
        model = arch(state)
        model.eval()
        self.model = model.to(self.device)
        if half:
            self.model.half()
        self.half = half
        return self

    def run(self, overlap: int = 0, interval: int = 0) -> VSGAN:
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
                interval=interval
            )
        )

        return self

    @torch.inference_mode()
    def execute(self, n: int, clip: vs.VideoNode, model: torch.nn.Module, half: bool = False, overlap: int = 0,
                interval: int = 0) -> vs.VideoNode:
        """
        Run the ESRGAN repo's Modified ESRGAN RRDBNet super-resolution code on a clip's frame.
        Unlike the original code, frames are modified directly as Tensors, without CV2.

        Thanks to VideoHelp for initial support, and @JoeyBallentine for his work on
        seamless chunk support.
        """

        # -- EGVSR Models

        if interval > 0:
            if str(n) not in self.tensor_cache:
                self.tensor_cache.clear()

                lr_images = [frame_to_tensor(clip.get_frame(n))]
                for i in range(1, interval):
                    if (n + i) >= clip.num_frames:
                        break
                    lr_images.append(frame_to_tensor(clip.get_frame(n + i)))
                lr_images = torch.stack(lr_images)
                lr_images = lr_images.unsqueeze(0)
                if half:
                    lr_images = lr_images.half()

                output, _, _, _, _ = model.forward_sequence(lr_images.to(self.device))
                output = output.squeeze(0)

                for i in range(output.shape[0]):  # interval
                    self.tensor_cache[str(n + i)] = output[i, :, :, :]

                del lr_images
                torch.cuda.empty_cache()

            return tensor_to_clip(clip, self.tensor_cache[str(n)])

        # -- ESRGAN Models

        lr_img = frame_to_tensor(clip.get_frame(n), half=half)
        lr_img.unsqueeze_(0)
        try:
            if not overlap:
                output_img = model(lr_img.to(self.device)).data
            elif overlap > 0:
                output_img = join_tiles(tuple(
                    self.model(tile_lr.to(self.device)).detach().cpu()
                    for tile_lr in tile_tensor(lr_img, overlap)
                ))
            else:
                raise ValueError("Invalid overlap. Must be a value greater than 0, or a False-y value to disable.")
        except RuntimeError as e:
            if "allocate" in str(e) or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
            raise

        return tensor_to_clip(clip, output_img)
