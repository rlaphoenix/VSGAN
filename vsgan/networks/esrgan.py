from __future__ import annotations

import functools
import uuid
from typing import Union

import torch
import vapoursynth as vs
from vapoursynth import core

from vsgan import archs
from vsgan.networks.basenetwork import BaseNetwork
from vsgan.utilities import frame_to_tensor, tensor_to_clip, tile_tensor_r


class ESRGAN(BaseNetwork):
    """
    ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
    By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
    and Chen Change Loy.

    Supported Models:
    - ESRGAN (old/new): https://arxiv.org/abs/1809.00219
    - ESRGAN+: https://arxiv.org/abs/2001.08073
    - Real-ESRGAN (v1/v2): https://arxiv.org/abs/2107.10833
    - A-ESRGAN: https://arxiv.org/abs/2112.10046
    """

    def __init__(self, clip: vs.VideoNode, *devices: Union[str, int]):
        super().__init__(clip, *devices)
        self.depth_cache: dict = {}

    def load(self, state: str) -> ESRGAN:
        """
        Load an ESRGAN model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Supported Model Files:
        - Must be a Generator model.
        - ESRGAN (old and new)
        - ESRGAN+
        - Real-ESRGAN (v1 and v2)
        - A-ESRGAN

        Parameters:
            state: Path to a supported PyTorch .pth Model state file.
        """
        state = torch.load(state)
        if "params" in state and "body.0.weight" in state["params"]:
            arch = archs.RealESRGANv2
        else:
            arch = archs.ESRGAN

        self._models.clear()

        for device in self._devices:
            model = arch(state)
            model.eval()
            self._models.append(model.to(device))

        return self

    def apply(self, overlap: int = 16) -> ESRGAN:
        """
        Apply the model on each frame of the clip.

        Overlap should generally be a multiple of 16. The larger the input resolution,
        the larger overlap may need to be set. Avoid using a value excessively large.

        Parameters:
            overlap: Amount to overlap each tile as to hide artefact seams.
        """
        if not self._models:
            raise ValueError("A model must be loaded before running.")

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self._models[0].scale,
                height=self.clip.height * self._models[0].scale
            ),
            functools.partial(
                self._apply,
                # must pass any argument that may change here, otherwise it will only use
                # the last change, even if you executed apply() before the change!
                id_=str(uuid.uuid4()),
                clip=self.clip,
                models=self._models,
                overlap_=overlap
            )
        )

        return self

    @torch.inference_mode()
    def _apply(
        self,
        n: int,
        id_: str,
        clip: vs.VideoNode,
        models: list[torch.nn.Module],
        overlap_: int
    ) -> vs.VideoNode:
        # split the workload evenly between n devices loaded at construction
        device_index = n % len(self._devices)

        # model's storage device should match chosen device, unless modified externally
        device = self._devices[device_index]
        model = models[device_index]

        lr_img = frame_to_tensor(clip.get_frame(n))\
            .to(device)\
            .clamp(0, 1)\
            .unsqueeze(0)

        if lr_img.dtype == torch.half:
            model.half()

        sr_img, depth = tile_tensor_r(lr_img, model, overlap_, self.depth_cache.get(id_))
        self.depth_cache[id_] = depth

        return tensor_to_clip(clip, sr_img)


__ALL__ = (ESRGAN,)
