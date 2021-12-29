from typing import Union

from vsgan.archs.EGVSR import EGVSR
from vsgan.archs.ESRGAN import ESRGAN, RealESRGANv2

MODEL_T = Union[ESRGAN, RealESRGANv2, EGVSR]

__ALL__ = (MODEL_T, ESRGAN, RealESRGANv2, EGVSR)
