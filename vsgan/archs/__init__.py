from typing import Union, OrderedDict

from torch import Tensor

from vsgan.archs.ESRGAN import ESRGAN
from vsgan.archs.RealESRGAN import RealESRGAN

MODEL_T = Union[ESRGAN, RealESRGAN]
STATE_T = OrderedDict[str, Tensor]

__ALL__ = (MODEL_T, STATE_T, ESRGAN, RealESRGAN)
