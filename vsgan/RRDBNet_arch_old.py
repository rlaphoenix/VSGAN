import math

import torch.nn as nn

from . import RRDBNet_arch_old_block as Block


class RRDBNet(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int, upscale: int = 4, norm_type=None,
                 act_type: str = 'leakyrelu', mode: str = 'CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
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
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        hr_conv0 = Block.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        hr_conv1 = Block.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = Block.sequential(
            fea_conv,
            Block.ShortcutBlock(Block.sequential(*rb_blocks, lr_conv)),
            *upsampler,
            hr_conv0,
            hr_conv1
        )

    def forward(self, x):
        x = self.model(x)
        return x
