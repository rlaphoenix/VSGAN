from __future__ import annotations

import functools
import uuid
from typing import Literal, Union

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan.archs import EGVSR as EGVSR_arch
from vsgan.networks.basenetwork import BaseNetwork
from vsgan.utilities import frame_to_tensor, tensor_to_clip


class EGVSR(BaseNetwork):
    """
    EGVSR - Efficient & Generic Video Super-Resolution.
    By Yanpeng Cao, Chengcheng Wang, Changjun Song, Yongming Tang, and He Li.
    https://arxiv.org/abs/2107.05307
    """

    def __init__(self, clip: vs.VideoNode, *devices: Union[str, int]):
        super().__init__(clip, *devices)
        self.tensor_cache: dict = {}

    def load(
        self,
        state: str,
        scale: int = 4,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 16,
        degradation: Literal["BI", "BD"] = "BI"
    ) -> EGVSR:
        """
        Load an EGVSR model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Parameters:
            state: Path to a supported PyTorch Model file.
            scale: Model Scale, the resulting scale relative to the input.
            in_nc: Input number of channels.
            out_nc: Output number of channels.
            nf: Number of filters.
            nb: Number of blocks.
            degradation: Upsample Function.
        """
        self._models.clear()

        for device in self._devices:
            model = EGVSR_arch(state, scale, in_nc, out_nc, nf, nb, degradation)
            model.eval()
            self._models.append(model.to(device))

        return self

    def apply(self, interval: int = 5) -> EGVSR:
        """
        Apply the model on each frame of the clip.

        Parameters:
            interval: Amount of frames ahead to inference. Must be greater than 0.
        """
        if not self._models:
            raise ValueError("A model must be loaded before running.")

        cycle = len(self._devices)

        if cycle > 1:
            clips = [
                core.std.SelectEvery(clip=self.clip, cycle=cycle, offsets=n)
                for n in range(cycle)
            ]
        else:
            clips = [self.clip]

        clips = [
            core.std.FrameEval(
                core.std.BlankClip(
                    clip=clip,
                    width=clip.width * self._models[0].scale,
                    height=clip.height * self._models[0].scale
                ),
                functools.partial(
                    self._apply,
                    # must pass any argument that may change here, otherwise it will only use
                    # the last change, even if you executed apply() before the change!
                    id_=str(uuid.uuid4()),
                    clip=clip,
                    device=self._devices[i],
                    model=self._models[i],
                    interval_=interval
                )
            )
            for i, clip in enumerate(clips)
        ]

        self.clip = core.std.Interleave(clips)

        return self

    @torch.inference_mode()
    def _apply(
        self,
        n: int,
        id_: str,
        clip: vs.VideoNode,
        device: torch.device,
        model: torch.nn.Module,
        interval_: int
    ) -> vs.VideoNode:
        frame_id = f"{id_}-{n}"
        if frame_id not in self.tensor_cache:
            self.tensor_cache.clear()  # don't keep unused frames in RAM

            lr_images = [frame_to_tensor(clip.get_frame(n))]
            for i in range(1, interval_):
                if (n + i) >= clip.num_frames:
                    break
                lr_images.append(frame_to_tensor(clip.get_frame(n + i)))

            lr_images = torch\
                .stack(lr_images)\
                .unsqueeze(0)\
                .to(device)\
                .clamp(0, 1)

            if lr_images.dtype == torch.half:
                # TODO: Fix EGVSR arch Half-precision support
                #       Currently has a weird form of grid effect, most likely a
                #       data type conversion mistake in relation to half-tensors
                # forcing back as full tensor for now
                lr_images = lr_images.to(torch.float32)
                # model.half()

            sr_images, _, _, _, _ = model.forward_sequence(lr_images)
            sr_images = sr_images.squeeze(0)
            for i in range(sr_images.shape[0]):  # interval
                self.tensor_cache[f"{id_}-{n + i}"] = tensor_to_clip(clip, sr_images[i, :, :, :])

        return self.tensor_cache[frame_id]


__ALL__ = (EGVSR,)
