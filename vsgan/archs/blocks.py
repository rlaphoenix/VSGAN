"""
Common Blocks between Architectures.
"""

from collections import OrderedDict
from typing import Literal, Optional, Union

import torch.nn as nn


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]] = 1,
    dilation: Union[int, tuple[int, ...]] = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: Optional[str] = "zero",
    norm_type: Optional[str] = None,
    act_type: Optional[str] = "relu",
    mode: Literal["CNA", "NAC", "CNAC"] = "CNA"
) -> nn.Sequential:
    """
    Convolution layer with Padding, Normalization, and Activation layers.
    mode: CNA: Conv -> Norm -> Act
          NAC: Norm -> Act  -> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    if mode not in ("CNA", "NAC", "CNAC"):
        raise ValueError(f"Unsupported Convolution mode: {mode}")

    # Padding layer
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    # Convolution layer
    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

    # Normalization layer
    n = norm(norm_type, in_nc if mode == "NAC" else out_nc) if norm_type else None

    # Activation layer
    a = act(
        act_type,
        # Important! inplace ReLU will modify the input, therefore wrong output
        # input----ReLU(inplace)----Conv--+----output
        #        |________________________|
        inplace=mode != "NAC" and norm_type is None
    ) if act_type else None

    if mode in ("CNA", "CNAC"):
        return sequential(p, c, n, a)

    if mode == "NAC":
        return sequential(n, a, p, c)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    """Flatten Sequential. It unwraps nn.Sequential."""
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ShortcutBlock(nn.Module):
    """Element-wise sum the output of a submodule to its input."""
    def __init__(self, submodule):
        super().__init__()
        self.sub = submodule

    def forward(self, x):
        return x + self.sub(x)

    def __repr__(self):
        return "Identity + \n|" + self.sub.__repr__().replace("\n", "\n|")


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type='zero', norm_type=None, act_type='relu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor ** 2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
