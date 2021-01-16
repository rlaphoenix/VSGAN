---
title: "Installation"
permalink: /installation/
excerpt: "Instructions for installing the theme for new and existing Jekyll based sites."
last_modified_at: 2021-01-16T12:16:00-00:00
toc: true
---

## Hardware Requirements

**GPU:** NVIDIA GPU that has support for CUDA 9.2+. A CUDA Compute Capability score of 6 or higher is recommended, and a score <= 2 will be incredibly slow, if it works at all.  
**CPU:** There's no specific requirement for the CPU, just try and use a CPU that won't bottle-neck your GPU. The CPU will be used, but not much.

**No Supported GPU?** If you don't mind waiting minutes or even hours per frame, then you can use your CPU instead of the GPU. Just note that it's not our responsibility if over-use of your CPU, lower it's life span, or even get it killed. Expect constantly high CPU Usage and Temperatures causing even your mouse to lag.

## Software Dependencies

1. [**Python**](https://python.org) 3.5+ and Python [PIP](https://pip.pypa.io/en/stable/installing).
2. [**VapourSynth**](https://vapoursynth.com) **latest recommended** or R40+. Ensure the Python version you have installed is supported by the version of VapourSynth that you will use. The supported Python versions may differ per OS.
3. [**PyTorch**](https://pytorch.org/get-started/locally) **latest recommended** or 1.6.0+. Ensure you install PyTorch with CUDA support if you plan to use your GPU.
4. [**NVIDIA CUDA**](https://developer.nvidia.com/cuda-downloads) **latest recommended** or 9.2+. Ensure the version is supported by the version of PyTorch that you will use.

**Tip:** Ensure Python is added to PATH when installed (Tick `Add Python X.X to PATH` in `Customize Installation` mode) for an optimal and swift installation experience.
{: .notice--info}

**Tip:** When installing VapourSynth, ensure you follow all the instructions on their official website for your OS. There are important instructions specific to each OS.
{: .notice--info}

## Installing VSGAN

After all that dependency installation you're finally going to get redemption for your efforts! :D

```bash
pip install vsgan
```
