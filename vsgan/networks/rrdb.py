from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.functional import pixel_unshuffle

from vsgan import blocks
from vsgan.constants import STATE_T


class RRDBNet(nn.Module):
    def __init__(
        self,
        state: STATE_T,
        norm: Optional[blocks.NORM_TYPES_T] = None,
        act: Optional[blocks.ACT_TYPES_T] = "leakyrelu",
        upsampler: Literal["upconv", "pixel_shuffle"] = "upconv",
        mode: blocks.CONV_MODE_T = "CNA"
    ) -> None:
        """
        Residual in Residual Dense Block Network.

        This is based on the old original ESRGAN code tagged `old-arch` as
        the latest code in the ESRGAN and BasicSR repositories have severely
        shot itself in the foot and removed some features like 1x scale support.

        Regardless, the code was updated to support PyTorch models trained on
        both the old and new network. Internally it will translate new model
        states to the old model state.

        Parameters:
            state: PyTorch Model State dictionary.
            norm: Normalization layer.
            act: Activation layer.
            upsampler: Upsample layer. I.e., "upconv", "pixel_shuffle".
            mode: Convolution mode.
        """
        super().__init__()

        self.state = state
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # wanted, possible key names
            # currently supports old, new, and newer RRDBNet arch models
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            "model.3.weight": ("upconv1.weight", "conv_up1.weight"),
            "model.3.bias": ("upconv1.bias", "conv_up1.bias"),
            "model.6.weight": ("upconv2.weight", "conv_up2.weight"),
            "model.6.bias": ("upconv2.bias", "conv_up2.bias"),
            "model.8.weight": ("HRconv.weight", "conv_hr.weight"),
            "model.8.bias": ("HRconv.bias", "conv_hr.bias"),
            "model.10.weight": ("conv_last.weight",),
            "model.10.bias": ("conv_last.bias",),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)"
            )
        }

        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())

        self.state: STATE_T = self.new_to_old_arch(self.state)
        self.in_nc = self.state["model.0.weight"].shape[1]
        self.out_nc = self.get_out_nc() or self.in_nc  # assume same as in nc if not found
        self.scale = self.get_scale()
        self.num_filters = self.state["model.0.weight"].shape[0]

        # detect if pixel-unshuffle was used (Real-ESRGAN)
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (self.in_nc / 4, self.in_nc / 16):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": blocks.upconv_block,
            "pixel_shuffle": blocks.pixel_shuffle_block
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError("Upsample mode [%s] is not found" % self.upsampler)

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    act_type=self.act
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = blocks.sequential(
            # fea conv
            blocks.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None
            ),
            blocks.ShortcutBlock(blocks.sequential(
                # rrdb blocks
                *[RRDB(
                    nc=self.num_filters,
                    kernel_size=3,
                    gc=32,
                    stride=1,
                    bias=True,
                    pad_type="zero",
                    norm_type=self.norm,
                    act_type=self.act,
                    mode="CNA",
                    plus=self.plus
                ) for _ in range(self.num_blocks)],
                # lr conv
                blocks.conv_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    kernel_size=3,
                    norm_type=self.norm,
                    act_type=None,
                    mode=self.mode
                )
            )),
            *upsample_blocks,
            # hr_conv0
            blocks.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act
            ),
            # hr_conv1
            blocks.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None
            )
        )

        # vapoursynth calls expect the real scale even if shuffled
        if self.shuffle_factor:
            self.scale = self.scale // self.shuffle_factor

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state: STATE_T) -> STATE_T:
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[f"model.1.sub./NB/.{kind}"]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        return old_state

    def get_out_nc(self) -> Optional[int]:
        max_part = 0
        out_nc = None
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > max_part:
                    max_part = part_num
                    out_nc = self.state[part].shape[0]
        return out_nc

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2 ** n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            x = pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
        return self.model(x)


# Blocks only used by RRDBNet

class ResidualDenseBlock5C(nn.Module):
    """
    5 Convolution Residual Dense Block.
    Residual Dense Network for Image Super-Resolution, CVPR 18.
    gc: growth channel, i.e. intermediate channels
    """

    def __init__(
        self,
        nc: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: Optional[blocks.PAD_TYPES_T] = "zero",
        norm_type: Optional[blocks.NORM_TYPES_T] = None,
        act_type: Optional[blocks.ACT_TYPES_T] = "leakyrelu",
        mode: blocks.CONV_MODE_T = "CNA",
        plus: bool = False
    ):
        super().__init__()
        last_act = None if mode == "CNA" else act_type

        self.conv1x1 = conv1x1(nc, gc) if plus else None
        self.conv1 = blocks.conv_block(
            nc, gc, kernel_size, stride,
            bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = blocks.conv_block(
            nc + gc, gc, kernel_size, stride,
            bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = blocks.conv_block(
            nc + 2 * gc, gc, kernel_size, stride,
            bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = blocks.conv_block(
            nc + 3 * gc, gc, kernel_size, stride,
            bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = blocks.conv_block(
            nc + 4 * gc, nc, 3, stride,
            bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(
        self,
        nc: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: Optional[blocks.PAD_TYPES_T] = "zero",
        norm_type: Optional[blocks.NORM_TYPES_T] = None,
        act_type: Optional[blocks.ACT_TYPES_T] = "leakyrelu",
        mode: blocks.CONV_MODE_T = "CNA",
        plus: bool = False
    ):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)
        self.rdb2 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)
        self.rdb3 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out.mul(0.2) + x


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


__ALL__ = (RRDBNet, ResidualDenseBlock5C, RRDB, conv1x1)
