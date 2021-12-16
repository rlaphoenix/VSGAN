import math

import torch.nn as nn

from vsgan.models.ESRGAN import Blocks as Block


class Network(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int, upscale: int = 4, norm_type=None,
                 act_type: str = 'leakyrelu', mode: str = 'CNA', upsample_mode='upconv'):
        """
        Residual in Residual Dense Block Network.

        This is specifically v0.1 (aka old-arch) and is not the newest revision code
        that's available at github:/xinntao/ESRGAN. This is on purpose, the newest
        code has hardcoded and severely limited the potential use of the Network.
        Specifically it has hardcoded the scale value to be `4` no matter what.

        :param in_nc: Input number of channels
        :param out_nc: Output number of channels
        :param nf: Number of filters
        :param nb: Number of blocks
        :param upscale: Scale relative to input
        :param norm_type: Normalization type
        :param act_type: Activation type
        :param mode: Convolution mode
        :param upsample_mode: Upsample block type. upconv, pixel_shuffle
        """
        super(Network, self).__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = Block.conv_block(in_nc=in_nc, out_nc=nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [Block.RRDB(
            nc=nf,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type='zero',
            norm_type=norm_type,
            act_type=act_type,
            mode='CNA'
        ) for _ in range(nb)]
        lr_conv = Block.conv_block(in_nc=nf, out_nc=nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = Block.upconv_block
        elif upsample_mode == 'pixel_shuffle':
            upsample_block = Block.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(in_nc=nf, out_nc=nf, upscale_factor=3, act_type=act_type)
        else:
            upsampler = [upsample_block(in_nc=nf, out_nc=nf, act_type=act_type) for _ in range(n_upscale)]

        hr_conv0 = Block.conv_block(in_nc=nf, out_nc=nf, kernel_size=3, norm_type=None, act_type=act_type)
        hr_conv1 = Block.conv_block(in_nc=nf, out_nc=out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = Block.sequential(
            fea_conv,
            Block.ShortcutBlock(Block.sequential(*rb_blocks, lr_conv)),
            *upsampler,
            hr_conv0,
            hr_conv1
        )

    def forward(self, x):
        return self.model(x)
