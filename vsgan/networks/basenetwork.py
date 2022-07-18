from __future__ import annotations

from abc import abstractmethod
from typing import Union

import torch
import vapoursynth as vs


class BaseNetwork:
    def __init__(self, clip: vs.VideoNode, *devices: Union[str, int]):
        """
        Create a PyTorch Device instance to use VSGAN with.
        It validates the supplied pytorch device identifier, and makes
        sure CUDA environment is available and ready.

        Parameters:
            clip: VapourSynth Video Node (aka clip). It must be RGB
                color-space.
            devices: One or more PyTorch device identifiers to use for the model.
                E.g., "cuda", "cuda:0", "cpu", 0, 1, and so on.
        """
        if not isinstance(clip, vs.VideoNode):
            raise ValueError(f"This is not a clip, {clip!r}")

        if not devices:
            raise ValueError("A Torch Device must be specified.")

        if clip.format.color_family != vs.RGB:
            raise ValueError("Only RGB clips are supported. RGB24 or RGBS recommended.")

        if any(str(x).lower() == "cpu" for x in devices):
            raise UserWarning(
                "VSGAN blocked an attempt to use your CPU as the torch device. "
                "Using your CPU will run it at very high utilisation and may lower its lifespan. "
                "If you are sure you would like to use your CPU, then use `cpu!`."
            )

        if any("cuda" in str(x) for x in devices) and not torch.cuda.is_available():
            raise EnvironmentError("CUDA Torch Device Specified but either NVIDIA CUDA or the device isn't available.")

        devices = [
            [x, "cpu"][str(x).lower() == "cpu!"]
            for x in devices
        ]

        self.clip: vs.VideoNode = clip
        self._devices: list[torch.device] = [
            torch.device(x)
            for x in devices
        ]
        self._models: list[torch.nn.Module] = []

    @abstractmethod
    def load(self, state: str) -> BaseNetwork:
        """
        Load a PyTorch model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Parameters:
            state: Path to a supported PyTorch .pth Model state file.
        """

    @abstractmethod
    def apply(self, overlap: int = 16) -> BaseNetwork:
        """
        Apply the model on each frame of the clip.

        Overlap should generally be a multiple of 16. The larger the input resolution,
        the larger overlap may need to be set. Avoid using a value excessively large.

        Parameters:
            overlap: Amount to overlap each tile as to hide artefact seams.
        """
