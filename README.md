# VSGAN

<p align="center">
<a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.6%2B-informational?style=flat-square" /></a>
<a href="https://github.com/rlaPHOENiX/VSGAN/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/rlaPHOENiX/VSGAN?style=flat-square"></a>
<a href="https://www.codefactor.io/repository/github/rlaPHOENiX/vsgan"><img src="https://www.codefactor.io/repository/github/rlaPHOENiX/vsgan/badge" alt="CodeFactor" /></a>
<a href="https://www.codacy.com/gh/rlaPHOENiX/VSGAN/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rlaPHOENiX/VSGAN&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/ff06331673f0459c9f3cc6443a7ac357"/></a>
<a href="https://github.com/rlaPHOENiX/VSGAN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/rlaPHOENiX/VSGAN?style=flat-square"></a>
<a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

# :page_facing_up: Introduction

This is a single image super-resolution generative adversarial network handler for VapourSynth.
Since VapourSynth will take the frames from a video, and feed it to VSGAN, it is essentially a single video super-resolution gan.
It is a direct port of [ESRGAN by xinntao](https://github.com/xinntao/ESRGAN), so all results, accomplishments, and such that ESRGAN does, VSGAN will do too.

Using the right pre-trained model, on the right image, can have tremendous results.  
Here's an example from a US Region 1 (NTSC) DVD of American Dad running with VSGAN (model not public)
![Example 1](examples/cmp_1.png)

# :camera: Qualitive Comparisons against other Super-Resolution Strategies

Following comparisons were taken from [ESRGAN's repo](https://github.com/xinntao/ESRGAN)
![qualitive1](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_01.jpg)
![qualitive2](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_02.jpg)
![qualitive3](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_03.jpg)
![qualitive4](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_04.jpg)

# Installation

Please install the following in order as-shown.

## 1. VapourSynth

*Note: VapourSynth **must** be installed before VSGAN. The package available to install from pip/pypi.org is a function wrapper for Python and still requires VapourSynth to be pre-installed. [[1]](http://vapoursynth.com/doc/installation.html#installation-via-pip-pypi)*

Official Installation Instructions are available for:

[Windows](http://vapoursynth.com/doc/installation.html#windows-installation-instructions), [Linux](http://vapoursynth.com/doc/installation.html#linux-installation-from-packages), [Mac OS X](http://vapoursynth.com/doc/installation.html#os-x-installation-from-packages)

## 2. mvsfunc

Mvsfunc is typically installed by default.

Somehow not installed?

**Windows:**

From within a command-prompt:

1. Run `where vsrepo.py` and copy the full path of vsrepo.py
2. Run `python "C:/path/to/vsrepo.py" install mvsfunc` (keeping the quotes around the path)

**Linux:**

You can most likely find it in your distro's package manager.

## 3. (optional, recommended) NVIDIA CUDA

If you plan to use VSGAN with your NVIDIA GPU as a device rather than your CPU (e.g. with torch-device strings: `0`, `1`, ..., or `cuda`) then NVIDIA CUDA must be installed for torch to be able to do so.

Using VSGAN on a CPU will heavily strain your system (most likely to even unusable amounts!) and will take *forever* to finish a single frame. I always recommend users of VSGAN to use a NVIDIA GPU with CUDA on a GTX 1050 or better. Just how fast is using a GPU compared to a CPU? A CPU could take minutes to do a single 720x480 frame x4 scale, whereas a GPU could take about half a second (on my GTX 1080ti).

There are far too many ways to install CUDA, many different ways on many different operating systems. Search up on ([DuckDuckGo](https://is.gd/Z4NpYy), [Google](http://google.com/search?q=nvidia+cuda+installation)) how to install it.

## 4. VSGAN

### Using PIP via PyPI (recommended)

Run `pip install vsgan` in terminal/cmd.

### Using PIP offline

*Note: there's no auto-updates or update notifications with this method. You may possibly be installing code that has not been tested as stable, that has not yet reached an actual update version, meaning you will be using the very-latest code which may be unstable.*

    git clone https://github.com/rlaPHOENiX/VSGAN.git && cd vsgan
    pip install .

If you don't have git nor wish to, you could also download the [master.zip](https://github.com/rlaPHOENiX/VSGAN/archive/master.zip), extract it, open a command prompt into that folder via `cd`, and `pip install .`

# Usage

Quick Example

    from vsgan import VSGAN

    # ...

    # create a VSGAN instance, and have it make a torch device with CUDA (or `"cpu"` or `0`, `1`, e.t.c)
    vsgan = VSGAN("cuda")
    # load a model into the VSGAN instance
    # you can run load_model() on the instance at any point to change the model
    vsgan.load_model(
        model=r"C:\User\PHOENiX\Documents\Models\PSNR_x4_DB.pth",
        scale=4
    )
    # use the VSGAN instance (with its loaded model) on a clip
    clip = vsgan.run(clip)

    # ...

    # don't forget to set the output in your vapoursynth script
    clip.set_output()

    # There's more you can do with load_model as well as run, see definitions below.

# Definitions

### VSGAN(device=[int/string])

Create a PyTorch Device instance using VSGAN for the provided device

*  `device` can be either a string or an int, acceptable values are "cpu", "cuda", and a device id number (e.g. 0, 1), default is "cuda" if available, otherwise it will use "cpu"

### VSGAN.load_model(model=[string], scale=[int], old_arch=[bool])

Load a model into the VSGAN Device instance

*  `model` must be a path to the .pth model. use r'' if the path has crazy characters like back-slashes `(\)`
*  `scale` must be an int > 0 and must match the model it was trained with
*  `old_arch` should only be used if the model does not work with the current arch or is a 1x scale model

### VSGAN.run(clip=clip, chunk=[bool])

This function takes the provided clip and runs every frame on [VSGAN.execute()](#vsgan.execute) as frames are being processed.

*  `chunk` if your system is running out of memory, try enable chunking as it will split the image into smaller sub-images and render them one by one, then finally merging them back together. Trading memory requirements for speed and accuracy. WARNING: Since the images will be processed separately, the result may have issues on the edges of the chunks, [an example of this issue](https://imgbox.com/g/Hht5NqKB0i)

## Internal use only

### VSGAN.execute(n=[int], clip=clip]

Executes the GAN model on n frame from clip.
