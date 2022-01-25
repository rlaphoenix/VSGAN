from __future__ import annotations

from typing import OrderedDict

import torch
import vapoursynth as vs
from torch import Tensor

VS_API = vs.__api_version__.api_major
IS_VS_API_4 = VS_API == 4

VS_DTYPE_MAP = {
    "RGB24": torch.int8,
    "RGB27": torch.int16,
    "RGB30": torch.int16,
    "RGB36": torch.int16,
    "RGB42": torch.int16,
    "RGB48": torch.int16,
    "RGBH": torch.float16,
    "RGBS": torch.float32,
    "GRAY8": torch.int8,
    "GRAY9": torch.int16,
    "GRAY10": torch.int16,
    "GRAY12": torch.int16,
    "GRAY14": torch.int16,
    "GRAY16": torch.int16,
    "GRAY32": torch.int32,
    "GRAYH": torch.float16,
    "GRAYS": torch.float32,
    "YUV410P8": torch.int8,
    "YUV411P8": torch.int8,
    "YUV420P8": torch.int8,
    "YUV420P9": torch.int16,
    "YUV420P10": torch.int16,
    "YUV420P12": torch.int16,
    "YUV420P14": torch.int16,
    "YUV420P16": torch.int16,
    "YUV422P8": torch.int8,
    "YUV422P9": torch.int16,
    "YUV422P10": torch.int16,
    "YUV422P12": torch.int16,
    "YUV422P14": torch.int16,
    "YUV422P16": torch.int16,
    "YUV440P8": torch.int8,
    "YUV444P8": torch.int8,
    "YUV444P9": torch.int16,
    "YUV444P10": torch.int16,
    "YUV444P12": torch.int16,
    "YUV444P14": torch.int16,
    "YUV444P16": torch.int16,
    "YUV444PH": torch.float16,
    "YUV444PS": torch.float32
}

STATE_T = OrderedDict[str, Tensor]
