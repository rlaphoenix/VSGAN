from __future__ import annotations

import functools

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan import archs
from vsgan.networks.basenetwork import BaseNetwork
from vsgan.utilities import frame_to_tensor, tensor_to_clip, recursive_tile_tensor


class ESRGAN(BaseNetwork):
    """
    ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
    By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
    and Chen Change Loy.
    """

    def load(self, model: str, half: bool = False) -> ESRGAN:
        """
        Load an ESRGAN model file and send to the PyTorch device.
        The model can be changed at any point.

        Supported Model Files:
        - Must be a Generator model.
        - ESRGAN (old and new)
        - ESRGAN+
        - Real-ESRGAN (v1 and v2)
        - A-ESRGAN

        Parameters:
            model: Path to a supported PyTorch Model file.
            half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
        """
        state = torch.load(model)
        if "params" in state and "body.0.weight" in state["params"]:
            arch = archs.RealESRGANv2
        else:
            arch = archs.ESRGAN
        model = arch(state)
        model.eval()
        self.model = model.to(self.device)
        if half:
            self.model.half()
        self.half = half
        return self

    @torch.inference_mode()
    def apply(self, overlap: int = 16) -> ESRGAN:
        """
        Apply the model on each frame of the clip.

        Overlap should generally be a multiple of 16. The larger the input resolution,
        the larger overlap may need to be set. Avoid using a value excessively large.

        Parameters:
            overlap: Amount to overlap each tile as to hide artefact seams.
        """
        if not self.model:
            raise ValueError("A model must be loaded before running.")

        def _apply(n: int, clip: vs.VideoNode, model: torch.nn.Module, half: bool, overlap_: int) -> vs.VideoNode:
            lr_img = frame_to_tensor(clip.get_frame(n), half=half)
            lr_img.unsqueeze_(0)
            output_img, depth = recursive_tile_tensor(lr_img.to(self.device), model, overlap_)
            return tensor_to_clip(clip, output_img)

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model.scale,
                height=self.clip.height * self.model.scale
            ),
            functools.partial(
                _apply,
                clip=self.clip,
                model=self.model,
                half=self.half,
                overlap_=overlap
            )
        )

        return self


__ALL__ = (ESRGAN,)
