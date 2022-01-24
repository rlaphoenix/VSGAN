from __future__ import annotations

import functools
from typing import Union

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan import archs
from vsgan.networks.basenetwork import BaseNetwork
from vsgan.utilities import frame_to_tensor, tensor_to_clip


class EGVSR(BaseNetwork):
    """
    EGVSR - Efficient & Generic Video Super-Resolution.
    By Yanpeng Cao, Chengcheng Wang, Changjun Song, Yongming Tang, and He Li.
    """

    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        super().__init__(clip, device)
        self.tensor_cache: dict = {}

    def load(
        self,
        model: str,
        scale: int = 4,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 16,
        degradation: str = "BI",
        half: bool = False
    ) -> EGVSR:
        """
        Load an EGVSR model file and send to the PyTorch device.
        The model can be changed at any point.

        Supported Model Files:
        - Must be a Generator model.
        - EGVSR only.

        Parameters:
            model: Path to a supported PyTorch Model file.
            scale: Model Scale, the resulting scale relative to the input.
            in_nc: Input number of channels.
            out_nc: Output number of channels.
            nf: Number of filters.
            nb: Number of blocks.
            degradation: Upsample Function.
            half: Reduce tensor accuracy from fp32 to fp16. Reduces VRAM, may improve speed.
        """
        model = archs.EGVSR(model, scale, in_nc, out_nc, nf, nb, degradation)
        model.eval()
        self.model = model.to(self.device)
        if half:
            self.model.half()
        self.half = half
        return self

    @torch.inference_mode()
    def apply(self, interval: int = 5) -> EGVSR:
        """
        Apply the model on each frame of the clip.

        Parameters:
            interval: Amount of frames ahead to inference. Must be greater than 0.
        """
        if not self.model:
            raise ValueError("A model must be loaded before running.")

        def _apply(n: int, clip: vs.VideoNode, model: torch.nn.Module, half: bool, interval_: int) -> vs.VideoNode:
            if str(n) not in self.tensor_cache:
                self.tensor_cache.clear()

                lr_images = [frame_to_tensor(clip.get_frame(n))]
                for i in range(1, interval_):
                    if (n + i) >= clip.num_frames:
                        break
                    lr_images.append(frame_to_tensor(clip.get_frame(n + i)))
                lr_images = torch.stack(lr_images)
                lr_images = lr_images.unsqueeze(0)
                if half:
                    lr_images = lr_images.half()

                output, _, _, _, _ = model.forward_sequence(lr_images.to(self.device))
                output = output.squeeze(0)

                for i in range(output.shape[0]):  # interval
                    self.tensor_cache[str(n + i)] = output[i, :, :, :]

                del lr_images
                torch.cuda.empty_cache()

            return tensor_to_clip(clip, self.tensor_cache[str(n)])

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model.scale,
                height=self.clip.height * self.model.scale
            ),
            functools.partial(
                _apply,
                clip=self.clip,
                model=self.model,
                half=self.half,
                interval_=interval
            )
        )

        return self


__ALL__ = (EGVSR,)
