import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan import IS_VS_API_4, MAX_DTYPE_VALUES


def frame_to_array(f: vs.VideoFrame) -> np.stack:
    """
    Convert a VapourSynth VideoFrame into a Numpy Array.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
    """
    return np.stack([
        np.asarray(f[plane])
        for plane in range(f.format.num_planes)
    ])


def frame_to_tensor(f: vs.VideoFrame, order: tuple[int, ...] = (0, 1, 2), clamp_zero=True, bgr2rgb=False,
                    normalize=False, half: bool = False) -> torch.Tensor:
    """
    Convert a VapourSynth VideoFrame into a PyTorch Tensor.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
        order: Change shape order to specified order. Default: CHW.
        clamp_zero: Clamp to 0,1 range.
        bgr2rgb: Flip Plane order from BGR to RGB. May be needed if loaded via OpenCV.
        normalize: Normalize (z-norm) from [0,1] range to [-1,1].
        half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
    """
    array = frame_to_array(f)

    if clamp_zero:
        max_val = MAX_DTYPE_VALUES.get(array.dtype, 1.0)
        array = array.astype(np.dtype("float32")) / max_val

    if order != (0, 1, 2) and len(order) == 3:
        array = np.transpose(array, order)

    array = torch.from_numpy(array).float()

    if half:
        array = array.half()

    if bgr2rgb:
        if array.shape[0] % 3 == 0:
            # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
            array = array.flip(-3)
        elif array.shape[0] == 4:
            # RGBA
            array = array[[2, 1, 0, 3], :, :]

    if normalize:
        array = ((array - 0.5) * 2.0).clamp(-1, 1)

    return array


def tensor_to_frame(f: vs.VideoFrame, t: torch.Tensor) -> vs.VideoFrame:
    """
    Copies each channel from a Tensor into a VapourSynth VideoFrame.
    Supports any depth and format, and will return in the same format.

    It expects the tensor array to have the dimension count (C) first
    in the shape, e.g., CHW or CWH.

    Parameters:
        f: VapourSynth frame to store retrieved planes.
        t: PyTorch Tensor array to retrieve planes from.
    """
    # TODO: - Is the return needed? Looks like in-place modification
    #       - What if the frame is read-only?

    array = t.squeeze(0).detach().clamp(0, 1).cpu().numpy()

    d_type = np.asarray(f[0]).dtype
    array = MAX_DTYPE_VALUES.get(d_type, 1.0) * array
    array = array.astype(d_type)

    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])

    return f


def tensor_to_clip(clip: vs.VideoNode, image: torch.Tensor) -> vs.VideoNode:
    """
    Convert a PyTorch Tensor into a VapourSynth VideoNode (clip).

    Expecting Torch shape to be in CHW order.

    Parameters:
        clip: Used to inherit expected return properties only.
        image: PyTorch Tensor.
    """
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
