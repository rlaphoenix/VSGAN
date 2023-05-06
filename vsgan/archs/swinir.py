from __future__ import annotations

import functools
import uuid
from typing import Optional, Union

import torch
import vapoursynth as vs
from torch import nn
from vapoursynth import core

from vsgan.archs.basearch import BaseArch
from vsgan.networks.swinir import SwinIR as SwinIR_Net
from vsgan.utilities import frame_to_tensor, tensor_to_clip, tile_tensor_r


class SwinIR(BaseArch):
    """
    SwinIR - Image Restoration Using Swin Transformer.
    By Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte.
    https://arxiv.org/abs/2108.10257
    """

    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        super().__init__(clip, device)
        self.depth_cache: dict = {}

    def load(
        self,
        state: str,
        img_size: Union[int, tuple[int, ...]] = 64,
        patch_size: Union[int, tuple[int, ...]] = 1,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False
    ) -> SwinIR:
        """
        Load a SwinIR model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Parameters:
            state: Path to a supported PyTorch Model file.
            img_size: Input image size.
            patch_size: Patch size.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_scale: Override default qk scale of head_dim ** -0.5 if set.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            ape: If True, add absolute position embedding to the patch embedding.
            patch_norm: If True, add normalization after patch embedding.
            use_checkpoint: Whether to use checkpointing to save memory.
        """
        state_dict = super().load(state)
        model = SwinIR_Net(state_dict, img_size, patch_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                           drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint)
        model.eval()
        self._model = model.to(self._device)
        return self

    def apply(self, overlap: int = 16) -> SwinIR:
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
            # SwinIR arch doesn't currently support float16, so convert lr tensor to float32
            lr_img = lr_img.float()

        sr_img, depth = tile_tensor_r(lr_img, model, overlap_, self.depth_cache.get(id_))
        self.depth_cache[id_] = depth

        return tensor_to_clip(clip, sr_img)


__ALL__ = (SwinIR,)
