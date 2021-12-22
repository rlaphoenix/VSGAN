from typing import Union

from vsgan.archs import ESRGAN, RealESRGAN
from vsgan.archs.ESRGAN import ESRGAN
from vsgan.archs.RealESRGAN import RealESRGAN

MODEL_T = Union[ESRGAN, RealESRGAN]

__ALL__ = (MODEL_T, ESRGAN, RealESRGAN)
