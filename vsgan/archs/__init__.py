from typing import Union

from vsgan.archs.EGVSR import EGVSR
from vsgan.archs.ESRGAN import ESRGAN

MODEL_T = Union[ESRGAN, EGVSR]

__ALL__ = (MODEL_T, ESRGAN, EGVSR)
