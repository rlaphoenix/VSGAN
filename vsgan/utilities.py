from __future__ import annotations

import gc

import numpy as np
import torch
import vapoursynth as vs
from torch import Tensor
from vapoursynth import core

from vsgan.constants import IS_VS_API_4, VS_DTYPE_MAP


def get_frame_plane(f: vs.VideoFrame, n: int) -> memoryview:
    """
    Get a VideoFrame's Plane data as a MemoryView or a numpy array.
    Supports VS API 3 and 4.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
        n: Plane number.
    """
    if IS_VS_API_4:
        return f[n]
    return f.get_read_array(n)  # type: ignore


def frame_to_tensor(f: vs.VideoFrame, as_f16=True) -> torch.Tensor:
    """
    Convert a VapourSynth VideoFrame into a PyTorch Tensor.

    Parameters:
        f: VapourSynth VideoFrame from a clip.
        as_f16: Convert to float16 in 0,1 range.
    """
    array = np.stack([
        np.asarray(
            get_frame_plane(f, plane),
            dtype=VS_DTYPE_MAP[f.format.name]
        )
        for plane in range(f.format.num_planes)
    ])

    if as_f16 and f.format.sample_type != vs.FLOAT:
        # convert to unsigned int -> float16 -> clamp to 0,1 range
        # 0,1 range is required for most model networks
        bps = f.format.bytes_per_sample * 8
        size = (2 ** f.format.bits_per_sample) - 1
        array = array.astype(np.dtype(f"uint{bps}"))
        # float16 is just enough for up to RGB48
        array = array.astype(np.dtype("float16")) / size

    tensor = torch.from_numpy(array)

    return tensor


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

    if f.format.sample_type != vs.FLOAT:
        # un-clamp back to full range for the format
        size = (2 ** f.format.bits_per_sample) - 1
        array = size * array

    array = array.astype(np.asarray(f[0]).dtype)

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


def tile_tensor(t: torch.Tensor, overlap: int = 16) -> tuple[torch.Tensor, ...]:
    """
    Tile PyTorch Tensor into 4 quadrants with an overlap between tiles.
    Expects input PyTorch Tensor's shape to end in HW order.
    """
    h, w = t.shape[-2:]

    top_left_lr = t[..., : h // 2 + overlap, : w // 2 + overlap]
    top_right_lr = t[..., : h // 2 + overlap, w // 2 - overlap:]
    bottom_left_lr = t[..., h // 2 - overlap:, : w // 2 + overlap]
    bottom_right_lr = t[..., h // 2 - overlap:, w // 2 - overlap:]

    return top_left_lr, top_right_lr, bottom_left_lr, bottom_right_lr


def tile_tensor_r(
    t: torch.Tensor,
    model: torch.nn.Module,
    overlap: int = 16,
    max_depth: int = None,
    current_depth: int = 1
) -> tuple[torch.Tensor, int]:
    """
    Recursively Tile PyTorch Tensor until the device has enough VRAM.
    It will try to tile as little as possible, and wont tile unless needed.
    Expects input PyTorch Tensor's shape to end in HW order.
    """
    if current_depth > 10:
        torch.cuda.empty_cache()
        gc.collect()
        raise RecursionError("Exceeded maximum tiling recursion of 10...")

    if max_depth is None or max_depth == current_depth:
        # attempt non-tiled super-resolution if no known depth, or at depth
        try:
            t_sr = model(t)
            return t_sr, current_depth
        except RuntimeError as e:
            if "allocate" in str(e) or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                gc.collect()  # TODO: Truly beneficial?
            else:
                raise

    # Not at known depth, and non-tiled super-resolution failed, try tiled
    tiles_lr = tile_tensor(t, overlap)

    # take depth from top_left result as the size would be same for all quadrants
    # by re-using the depth, we can know exactly how much tiling is needed immediately
    tiles_lr_top_left, depth = tile_tensor_r(tiles_lr[0], model, overlap, current_depth=current_depth + 1)
    tiles_lr_top_right, _ = tile_tensor_r(tiles_lr[1], model, overlap, depth, current_depth=current_depth + 1)
    tiles_lr_bottom_left, _ = tile_tensor_r(tiles_lr[2], model, overlap, depth, current_depth=current_depth + 1)
    tiles_lr_bottom_right, _ = tile_tensor_r(tiles_lr[3], model, overlap, depth, current_depth=current_depth + 1)

    output_img = join_tiles(
        (tiles_lr_top_left, tiles_lr_top_right, tiles_lr_bottom_left, tiles_lr_bottom_right),
        overlap * model.scale
    )

    return output_img, depth


def join_tiles(tiles: tuple[torch.Tensor, ...], overlap: int) -> torch.Tensor:
    """
    Join Tiled PyTorch Tensor quadrants into one large PyTorch Tensor.
    Expects input PyTorch Tensor's shapes to end in HW order.

    Ensure the overlap value is what it currently is, possibly after
    super-resolution, not before!

    Parameters:
        tiles: The PyTorch Tensor tiles you wish to rejoin.
        overlap: The amount of overlap currently between tiles.
    """
    b, c, h, w = tiles[0].shape

    h = (h - overlap) * 2
    w = (w - overlap) * 2

    joined_tile = torch.empty((b, c, h, w), dtype=tiles[0].dtype, device=tiles[0].device)
    joined_tile[..., : h // 2, : w // 2] = tiles[0][..., : h // 2, : w // 2]
    joined_tile[..., : h // 2, -w // 2:] = tiles[1][..., : h // 2, -w // 2:]
    joined_tile[..., -h // 2:, : w // 2] = tiles[2][..., -h // 2:, : w // 2]
    joined_tile[..., -h // 2:, -w // 2:] = tiles[3][..., -h // 2:, -w // 2:]

    return joined_tile


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    Chunk a Tensor to batches of n windows/chunks.

    Parameters:
        x: Tensor in the shape (b, h, w, c)
        window_size: Window size

    Returns Tensor in the shape (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """
    Stitch back together batches of Windows/Chunks.

    Parameters:
        windows: Tensor in the shape (num_windows*b, window_size, window_size, c)
        window_size: Window size
        h: Height of image
        w: Width of image

    Returns stitched tensor in the shape (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x
