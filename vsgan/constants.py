from __future__ import annotations

from typing import OrderedDict

import numpy as np
import vapoursynth as vs
from torch import Tensor

VS_API = vs.__api_version__.api_major
IS_VS_API_4 = VS_API == 4

VS_DTYPE_MAP = {
    "RGB24": np.int8,
    "RGB27": np.int16,
    "RGB30": np.int16,
    "RGB36": np.int16,
    "RGB42": np.int16,
    "RGB48": np.int16,
    "RGBH": np.float16,
    "RGBS": np.float32,
    "GRAY8": np.int8,
    "GRAY9": np.int16,
    "GRAY10": np.int16,
    "GRAY12": np.int16,
    "GRAY14": np.int16,
    "GRAY16": np.int16,
    "GRAY32": np.int32,
    "GRAYH": np.float16,
    "GRAYS": np.float32,
    "YUV410P8": np.int8,
    "YUV411P8": np.int8,
    "YUV420P8": np.int8,
    "YUV420P9": np.int16,
    "YUV420P10": np.int16,
    "YUV420P12": np.int16,
    "YUV420P14": np.int16,
    "YUV420P16": np.int16,
    "YUV422P8": np.int8,
    "YUV422P9": np.int16,
    "YUV422P10": np.int16,
    "YUV422P12": np.int16,
    "YUV422P14": np.int16,
    "YUV422P16": np.int16,
    "YUV440P8": np.int8,
    "YUV444P8": np.int8,
    "YUV444P9": np.int16,
    "YUV444P10": np.int16,
    "YUV444P12": np.int16,
    "YUV444P14": np.int16,
    "YUV444P16": np.int16,
    "YUV444PH": np.float16,
    "YUV444PS": np.float32
}

STATE_T = OrderedDict[str, Tensor]
