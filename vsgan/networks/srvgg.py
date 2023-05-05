from __future__ import annotations

import math
from typing import Optional

from torch import nn
from torch.nn.functional import interpolate

from vsgan import blocks as block
from vsgan.constants import STATE_T


class SRVGGNetCompact(nn.Module):
    def __init__(self, state: STATE_T, act_type: Optional[block.ACT_TYPES_T] = "prelu") -> None:
        """
        A compact VGG-style network structure for super-resolution.

        It is a compact network structure, which performs upsampling in the last layer and no
        convolution is conducted on the HR feature space.

        This is specifically the alternate version used mainly for video inference.
        However, this is still not a video super-resolution network as it still only takes in
        one frame (or image) at a time. See:
        https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md#for-anime-videos

        Parameters:
            state: PyTorch Model State dictionary.
            act_type: Activation type. I.e., "relu", "prelu", "leakyrelu".
        """
        super().__init__()

        self.state = state
        self.act_type = act_type

        self.state_keys = list(self.state.keys())

        self.num_in_ch = self.get_in_nc()
        self.num_out_ch = self.num_in_ch  # TODO: Find a way. Assuming same as in_nc...
        self.num_conv = self.get_num_conv()
        self.num_feat = self.get_num_feats()
        self.pixel_shuffle_shape = self.get_pixel_shuffle_shape()
        self.scale = self.get_scale()

        # body structure
        self.body = nn.ModuleList()
        for n in range(self.num_conv + 1):
            self.body.append(nn.Conv2d(self.num_in_ch if n == 0 else self.num_feat, self.num_feat, 3, 1, 1))
            self.body.append({
                "relu": nn.ReLU(inplace=True),
                "prelu": nn.PReLU(num_parameters=self.num_feat),
                "leakyrelu": nn.LeakyReLU(negative_slope=0.1, inplace=True)
            }[self.act_type])

        # last conv
        self.body.append(nn.Conv2d(self.num_feat, self.pixel_shuffle_shape, 3, 1, 1))

        # upsampler
        self.upsampler = nn.PixelShuffle(self.scale)

        self.load_state_dict(self.state, strict=False)

    def get_in_nc(self) -> int:
        return self.state[self.state_keys[0]].shape[1]

    def get_num_conv(self) -> int:
        return (int(self.state_keys[-1].split(".")[1]) - 2) // 2

    def get_num_feats(self) -> int:
        return self.state[self.state_keys[0]].shape[0]

    def get_pixel_shuffle_shape(self) -> int:
        return self.state[self.state_keys[-1]].shape[0]

    def get_scale(self) -> int:
        # Assume out_nc is the same as in_nc
        # I cant think of a better way to do that
        self.num_out_ch = self.num_in_ch
        scale = math.sqrt(self.pixel_shuffle_shape / self.num_out_ch)
        if scale - int(scale) > 0:
            print("out_nc is probably different than in_nc, scale calculation might be wrong")
        scale = int(scale)
        return scale

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = interpolate(x, scale_factor=self.scale, mode="nearest")
        out += base
        return out


__ALL__ = (SRVGGNetCompact,)
