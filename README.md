# VSGAN

VapourSynth Single Image Super-Resolution Generative Adversarial Network (GAN)

[![Python Version](https://img.shields.io/badge/python-3.6%2B-informational?style=flat)](https://python.org)
[![License](https://img.shields.io/github/license/rlaPHOENiX/VSGAN?style=flat)](https://github.com/rlaPHOENiX/VSGAN/blob/master/LICENSE)
[![Codacy](https://app.codacy.com/project/badge/Grade/ff06331673f0459c9f3cc6443a7ac357)](https://codacy.com/gh/rlaPHOENiX/VSGAN/dashboard?utm_source=github.com&utm_medium=referral&utm_content=rlaPHOENiX/VSGAN&utm_campaign=Badge_Grade)
[![Issues](https://img.shields.io/github/issues/rlaPHOENiX/VSGAN?style=flat)](https://github.com/rlaPHOENiX/VSGAN/issues)
[![PR's Accepted](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://makeapullrequest.com)

## Introduction

This is a single image super-resolution generative adversarial network handler for VapourSynth.
Since VapourSynth will take the frames from a video, and feed it to VSGAN, it is essentially a single video super-resolution gan.
It is a direct port of [ESRGAN by xinntao](https://github.com/xinntao/ESRGAN), so all results, accomplishments, and such that ESRGAN does, VSGAN will do too.

Using the right pre-trained model, on the right image, can have tremendous results.  
Here's an example from a US Region 1 (NTSC) DVD of American Dad running with VSGAN (model not public)
![Example 1](examples/cmp_1.png)

## Qualitive Comparisons against other Super-Resolution Strategies

Following comparisons were taken from [ESRGAN's repo](https://github.com/xinntao/ESRGAN)
![qualitive1](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_01.jpg)
![qualitive2](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_02.jpg)
![qualitive3](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_03.jpg)
![qualitive4](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_04.jpg)

## Installation

### Requirements

1.  NVIDIA GPU that has support for CUDA 9.2+. A CUDA Compute Capability score of 6 or higher is recommended, and a score &lt;= 2 will be incredibly slow, if it works at all.
2.  CPU that isn't from the stone age. While this is going to do 90% of stuff on the GPU, a super bottle-knecking CPU could limit you're GPU's potential.
3.  An ESRGAN model file to use. Either train one or get an already trained one. There's new models being trained every day in all kinds of communities, with all kinds of specific purposes for each model, like denoising, upscaling, cleaning, inpainting, b&w to color, e.t.c. You can find models on the [Game Upscale Discord](https://discord.gg/cpAUpDK) or their [Upscale.wiki Model Database](https://upscale.wiki/wiki/Model_Database). The model database may not be as active as the Discord though.

### Dependencies

Install dependencies in the listed order:

1.  [Python](https://python.org) 3.6+ and [pip](https://pip.pypa.io/en/stable/installing). The required pip packages are listed in the [requirements.txt](https://github.com/rlaPHOENiX/VSMPEG/blob/master/requirements.txt) file.
2.  [VapourSynth](https://vapoursynth.com). Ensure the Python version you have installed is supported by the version you are installing. The supported Python versions may differ per OS.
3.  [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).
4.  [PyTorch](https://pytorch.org/get-started/locally) 1.6.0+, latest version is _always_ recommended.

#### Important information when Installing Python, VapourSynth, and PyTorch

1.  Ensure the Python version you have installed (or are going to install) is supported by the version of VapourSynth and PyTorch you are installing. The supported Python versions in correlation to a VapourSynth or PyTorch version may differ per OS, noticeably on Windows due to it's Python environment in general.
2.  When installing Python and VapourSynth, you will be given the option to "Install for all users" by both. Make sure your chosen answer matches for both installations or VapourSynth and Python wont be able to find each other.

Important note for Windows users: It is very important for you to tick the checkbox "Add Python X.X to PATH" while installing. The Python installer's checkbox that states "Install launcher for all users" is not referring to the Python binaries. To install for all users you must click "Customize installation" and in there, after "Optional Features" section, it will have a checkbox titled "Install for all users" unticked by default so tick it.

#### Tips on Installing PyTorch

Go to the [Get Started Locally page](https://pytorch.org/get-started/locally) and choose the following options:

```
PyTorch Build: `Stable`  
Package: `Pip`
Language: `Python`
CUDA: Latest available version, must match the installed version.
```

Then run the command provided by the `Run this Command:` text field.

#### Tips on Installing NVIDIA CUDA

Go to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads) and download and install the version you selected on the PyTorch page earlier.

If you chose for example `11.0` then `11.0` and >= `11.0` versions should work, but if you chose for example `10.2`, then chances are you need specifically version `10.2`, and not > `10.2`. However, I cannot confirm if this is the case.

### Finally, Installing VSGAN

It's as simple as running `pip install vsgan`

## Usage (Quick Example)

```py
from vapoursynth import RGB24
from vsgan import VSGAN

# ...

# Create a VSGAN instance, which creates a pytorch device instance
vsgan = VSGAN("cuda")  # available options: `"cuda"`, `0`, `1`, ..., e.t.c
# Load an ESRGAN model into the VSGAN instance
# Tip: You can run load_model() at any point to change the model
vsgan.load_model(r"C:\Users\PHOENiX\Documents\ESRGAN Models\PSNR_x4_DB.pth")
# Convert the clip to RGB24 as ESRGAN can only work with linear RGB data
clip = core.resize.Point(clip, format=RGB24)  # RGB24 is a int constant that was imported earlier
# Use the VSGAN instance (with its loaded model) on a clip
clip = vsgan.run(clip)
# Convert back to any other color space if you wish.

# ...

# don't forget to set the output in your vapoursynth script
clip.set_output()
```

## Documentation

### VSGAN(\[device: Union[int,str]="cuda"])

Create a PyTorch Device instance using VSGAN for the provided device

-   `device`: Acceptable values are `"cuda"`, and a device id number (e.g. `0`, `1`). `"cpu"` is not allowed as it's simply too slow and I don't want people hurting their CPU's.

### VSGAN.load_model(model: str)

Load a model into the VSGAN Device instance

-   `model`: A path to an ESRGAN .pth model file.

### VSGAN.run(clip: VideoNode[, chunk: bool=False])

Executes VSGAN on the provided clip, returning the resulting in a new clip.

-   `clip`: Clip to use as the input frames. It must be RGB. It will also return as RGB.
-   `chunk`: If your system is running out of memory, try enable `chunk` as it will split the image into smaller sub-images and render them one by one, then finally merging them back together. Trading memory requirements for speed and accuracy. WARNING: Since the images will be processed separately, the result may have issues on the edges of the chunks, [an example of this issue](https://imgbox.com/g/Hht5NqKB0i).

### VSGAN.execute(n: int, clip: VideoNode]

Executes the GAN model on `n`th frame from `clip`.

-   `n`: Frame number.
-   `clip`: Clip to get the frame from.
