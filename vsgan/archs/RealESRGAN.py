from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

from vsgan.constants import STATE_T


class RealESRGAN(nn.Module):
    def __init__(self, model: str, scale: int, in_nc: int = 3, out_nc: int = 3, num_feat: int = 64, num_block: int = 23,
                 num_grow_ch: int = 32) -> None:
        """
        Real-ESRGAN - Practical Algorithms for General Image Restoration.
        By Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan.

        We extend ESRGAN for scale x2 and scale x1.
        Note: This is one option for scale 1, scale 2 in ESRGAN.
        We first employ the pixel-unshuffle (an inverse operation of
        pixelshuffle to reduce the spatial size and enlarge the channel size
        before feeding inputs into the main ESRGAN architecture.

        Args:
            in_nc: Number of input channels
            out_nc: Number of output channels
            num_feat: Number of intermediate features
            num_block: Block number in the trunk network
            num_grow_ch: Number of channels for each growth
        """
        super(RealESRGAN, self).__init__()

        self.model = model
        self.scale = scale
        self.state: STATE_T = torch.load(self.model, map_location="cpu")["params_ema"]

        if scale == 2:
            in_nc = in_nc * 4
        elif scale == 1:
            in_nc = in_nc * 16

        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.load_state_dict(self.state)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(functional.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


@torch.no_grad()
def default_init_weights(module_list: Union[list[nn.Module], nn.Module], scale: float = 1, bias_fill: float = 0,
                         **kwargs) -> None:
    """
    Initialize network weights.

    Args:
        module_list: Modules to be initialized.
        scale: Scale initialized weights, especially for residual blocks.
        bias_fill: The value to fill bias.
        kwargs: Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        """
        Residual Dense Block.

        Args:
            num_feat: Channel number of intermediate features.
            num_grow_ch: Channels for each growth.
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        """
        Residual in Residual Dense Block.

        Args:
            num_feat: Channel number of intermediate features.
            num_grow_ch: Channels for each growth.
        """
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    """
    Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    if hh % scale != 0 or hw % scale != 0:
        raise ValueError("Shape was not evenly divisible by scale")
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
