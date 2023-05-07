from __future__ import annotations

import functools
import uuid
from typing import Union

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.archs.basearch import BaseArch
from vsgan.networks.rrdb import RRDBNet
from vsgan.networks.srvgg import SRVGGNetCompact
from vsgan.utilities import frame_to_tensor, tensor_to_clip, tile_tensor_r


class ESRGAN(BaseArch):
    """
    ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
    By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
    and Chen Change Loy.

    Supports the following iterative architectures/degradation models:
    - ESRGAN (old/new): https://arxiv.org/abs/1809.00219
    - ESRGAN+: https://arxiv.org/abs/2001.08073
    - BSRGAN: https://arxiv.org/abs/2103.14006
    - Real-ESRGAN (v1 only): https://arxiv.org/abs/2107.10833
    - A-ESRGAN: https://arxiv.org/abs/2112.10046
    """

    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        super().__init__(clip, device)
        self.depth_cache: dict = {}

    def load(self, state: str) -> ESRGAN:
        """
        Load an ESRGAN model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Supported Model Files:
        - Must be a Generator model.
        - ESRGAN (old and new)
        - ESRGAN+
        - Real-ESRGAN (v1 and v2)
        - A-ESRGAN

        Parameters:
            state: Path to a supported PyTorch .pth Model state file.
        """
        state_dict = super().load(state)
        if "body.0.weight" in state_dict:
            arch = SRVGGNetCompact
        else:
            arch = RRDBNet
        model = arch(state_dict)
        model.eval()
        self._model = model.to(self._device)
        return self

    def apply(self, overlap: int = 16) -> ESRGAN:
        """
        Apply the model on each frame of the clip.

        Overlap should generally be a multiple of 16. The larger the input resolution,
        the larger overlap may need to be set. Avoid using a value excessively large.

        Parameters:
            overlap: Amount to overlap each tile as to hide artefact seams.
        """
        if not self._model:
            raise ValueError("A model must be loaded before running.")

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self._model.scale,
                height=self.clip.height * self._model.scale
            ),
            functools.partial(
                self._apply,
                id_=str(uuid.uuid4()),
                clip=self.clip,
                model=self._model,
                overlap_=overlap
            )
        )

        return self

    @torch.inference_mode()
    def _apply(self, n: int, id_: str, clip: vs.VideoNode, model: torch.nn.Module, overlap_: int) -> vs.VideoNode:
        lr_img = frame_to_tensor(clip.get_frame(n))\
            .to(self._device)\
            .clamp(0, 1)\
            .unsqueeze(0)

        if lr_img.dtype == torch.half:
            model.half()

        sr_img, depth = tile_tensor_r(lr_img, model, overlap_, self.depth_cache.get(id_))
        self.depth_cache[id_] = depth

        return tensor_to_clip(clip, sr_img)


__ALL__ = (ESRGAN,)
