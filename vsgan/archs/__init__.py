from typing import Union, OrderedDict

from torch import Tensor

from vsgan.archs.ESRGAN import ESRGAN

MODEL_T = ESRGAN
STATE_T = OrderedDict[str, Tensor]

__ALL__ = (MODEL_T, STATE_T, ESRGAN)
