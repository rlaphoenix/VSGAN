"""
Common Blocks between Architectures.
"""

from typing import Literal, Optional, Union

import torch.nn as nn

CONV_MODE_T = Literal["CNA", "NAC", "CNAC"]
PAD_TYPES_T = Literal["reflect", "replicate", "zero"]
NORM_TYPES_T = Literal["batch", "instance"]
ACT_TYPES_T = Literal["relu", "leakyrelu", "prelu"]


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]] = 1,
    dilation: Union[int, tuple[int, ...]] = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: Optional[PAD_TYPES_T] = "zero",
    norm_type: Optional[NORM_TYPES_T] = None,
    act_type: Optional[ACT_TYPES_T] = "relu",
    mode: CONV_MODE_T = "CNA"
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


def act(
    act_type: ACT_TYPES_T, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1
) -> Union[nn.ReLU, nn.LeakyReLU, nn.PReLU]:
    """
    Helper for creating an Activation layer.

    Parameters:
        act_type: The Activation layer function to use.
        inplace: Do the operation in-place, used for 'relu' and 'leakyrelu'.
        neg_slope: Controls the angle of the negative slope, used for 'leakyrelu'.
            Also used as the initial value in 'prelu'.
        n_prelu: Number of parameters to learn, used for 'prelu'.
    """
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f"Activation layer [{act_type}] is not supported.")


def norm(norm_type: NORM_TYPES_T, nc: int) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    """
    Helper for creating a Normalization layer.

    Parameters:
        norm_type: The Normalization layer function to use.
        nc: The number of channels.
    """
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(nc, affine=False)
    raise NotImplementedError(f"Normalization layer [{norm_type}] is not supported.")


def pad(pad_type: PAD_TYPES_T, padding: int) -> Union[nn.ReflectionPad2d, nn.ReplicationPad2d, None]:
    """
    Helper for creating a Padding layer.

    Parameters:
        pad_type: The Padding layer function to use.
        padding: The padding amount.

    Returns no layer if padding amount is 0.
    """
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        return nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        return nn.ReplicationPad2d(padding)
    raise NotImplementedError(f"Padding layer [{pad_type}] is not supported.")


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args: Union[nn.Sequential, nn.Module]) -> nn.Sequential:
    """
    Flatten inputs into a single Sequential in the order provided.
    Any input that is not a Sequential or Module will be discarded.
    """
    modules = []
    for arg in args:
        if isinstance(arg, nn.Sequential):
            for submodule in arg.children():
                modules.append(submodule)
        elif isinstance(arg, nn.Module):
            modules.append(arg)

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
