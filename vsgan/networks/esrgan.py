from __future__ import annotations

import functools
from typing import Union

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

    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        super().__init__(clip, device)
        self.depth_cache: dict = {}

    def load(self, model: str) -> ESRGAN:
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
        """
        state = torch.load(model)
        if "params" in state and "body.0.weight" in state["params"]:
            arch = archs.RealESRGANv2
        else:
            arch = archs.ESRGAN
        model = arch(state)
        model.eval()
        self.model = model.to(self.device)
        return self

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

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model.scale,
                height=self.clip.height * self.model.scale
            ),
            functools.partial(
                self._apply,
                i=str(len(self.depth_cache)),
                clip=self.clip,
                model=self.model,
                overlap_=overlap
            )
        )

        return self

    @torch.inference_mode()
    def _apply(self, n: int, i: str, clip: vs.VideoNode, model: torch.nn.Module, overlap_: int) -> vs.VideoNode:
        lr_img = frame_to_tensor(clip.get_frame(n))
        lr_img.unsqueeze_(0)
        lr_img = lr_img.to(self.device)

        if lr_img.dtype == torch.half:
            model.half()

        sr_img, depth = recursive_tile_tensor(
            t=lr_img,
            model=model,
            overlap=overlap_,
            max_depth=self.depth_cache.get(i)
        )
        self.depth_cache[i] = depth

        return tensor_to_clip(clip, sr_img)


__ALL__ = (ESRGAN,)
