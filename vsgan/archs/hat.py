from __future__ import annotations

import functools
import uuid
from typing import Optional, Union

import torch
import vapoursynth as vs
from torch import nn
from vapoursynth import core

from vsgan.archs.basearch import BaseArch
from vsgan.networks.hat import HAT as HAT_Net
from vsgan.utilities import frame_to_tensor, tensor_to_clip, tile_tensor_r


class HAT(BaseArch):
    """
    Hybrid Attention Transformer - Activating More Pixels in Image Super-Resolution Transformer.
    By Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong.
    https://arxiv.org/abs/2205.04437
    """

    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        super().__init__(clip, device)
        self.depth_cache: dict = {}

    def load(
        self,
        state: str,
        img_size: Union[int, tuple[int, int]] = 64,
        patch_size: Union[int, tuple[int, ...]] = 1,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        overlap_ratio: float = 0.5,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        img_range: float = 1.
    ) -> HAT:
        """
        Load a SwinIR model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Parameters:
            state: PyTorch Model State dictionary.
            img_size (int | tuple(int)): Input image size. Default 64
            patch_size (int | tuple(int)): Patch size. Default: 1
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate: Dropout rate. Default: 0
            attn_drop_rate: Attention dropout rate. Default: 0
            drop_path_rate: Stochastic depth rate. Default: 0.1
            norm_layer: Normalization layer. Default: nn.LayerNorm.
            ape: If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm: If True, add normalization after patch embedding. Default: True
            use_checkpoint: Whether to use checkpointing to save memory. Default: False
            img_range: Image range. 1. or 255.
        """
        state_dict = super().load(state)
        model = HAT_Net(state_dict, img_size, patch_size, compress_ratio, squeeze_factor, conv_scale,
                        overlap_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                        norm_layer, ape, patch_norm, use_checkpoint, img_range)
        model.eval()
        self._model = model.to(self._device)
        return self

    def apply(self, overlap: int = 16) -> HAT:
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
            # HAT arch doesn't currently support float16, so convert lr tensor to float32
            lr_img = lr_img.float()

        sr_img, depth = tile_tensor_r(lr_img, model, overlap_, self.depth_cache.get(id_))
        self.depth_cache[id_] = depth

        return tensor_to_clip(clip, sr_img)


__ALL__ = (HAT,)
