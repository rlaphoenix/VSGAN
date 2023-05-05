from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

import torch
import vapoursynth as vs


class BaseArch:
    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        """
        Create a PyTorch Device instance to use VSGAN with.
        It validates the supplied pytorch device identifier, and makes
        sure CUDA environment is available and ready.

        Parameters:
            clip: VapourSynth Video Node (aka clip). It must be RGB
                color-space.
            device: PyTorch device identifier to use for the model. E.g.,
                "cuda", "cpu", 0, 1, and so on.
        """
        if not isinstance(clip, vs.VideoNode):
            raise ValueError(f"This is not a clip, {clip!r}")

        if clip.format.color_family != vs.RGB:
            raise ValueError("Only RGB clips are supported. RGB24 or RGBS recommended.")

        device = device.strip().lower() if isinstance(device, str) else device
        if device == "":
            raise ValueError("Device must be provided.")
        if device == "cpu":
            raise UserWarning(
                "VSGAN blocked an attempt to use your CPU as the torch device. "
                "Using your CPU will run it at very high utilisation and may lower its lifespan. "
                "If you are sure you would like to use your CPU, then use `cpu!`."
            )
        if device == "cpu!":
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise EnvironmentError("Either NVIDIA CUDA or the device (%s) isn't available." % device)

        self.clip: vs.VideoNode = clip
        self._device: torch.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None

    @abstractmethod
    def load(self, state: str) -> BaseArch:
        """
        Load a PyTorch model state file and send to the PyTorch device.
        The model state can be changed at any point.

        Parameters:
            state: Path to a supported PyTorch .pth Model state file.
        """

    @abstractmethod
    def apply(self, overlap: int = 16) -> BaseArch:
        """
        Apply the model on each frame of the clip.

        Overlap should generally be a multiple of 16. The larger the input resolution,
        the larger overlap may need to be set. Avoid using a value excessively large.

        Parameters:
            overlap: Amount to overlap each tile as to hide artefact seams.
        """


__ALL__ = (BaseArch,)
