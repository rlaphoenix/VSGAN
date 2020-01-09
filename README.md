# VSGAN

<p align="center">
<a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.6%2B-informational?style=flat-square" /></a>
<a href="https://github.com/rlaPHOENiX/VSGAN/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/rlaPHOENiX/VSGAN?style=flat-square"></a>
<a href="https://www.codefactor.io/repository/github/rlaPHOENiX/vsgan"><img src="https://www.codefactor.io/repository/github/rlaPHOENiX/vsgan/badge" alt="CodeFactor" /></a>
<a href="https://www.codacy.com/manual/rlaPHOENiX/VSGAN?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rlaPHOENiX/VSGAN&amp;utm_campaign=Badge_Grade"><img src="https://api.codacy.com/project/badge/Grade/1c7d12d0b4334efaa30c37eec3251b6a"/></a>
<a href="https://github.com/rlaPHOENiX/VSGAN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/rlaPHOENiX/VSGAN?style=flat-square"></a>
<a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

## :page_facing_up: Introduction

This is a single image super-resolution generative adversarial network handler for VapourSynth.
Since VapourSynth will take the frames from a video, and feed it to VSGAN, it is essentially a single video super-resolution gan.
It is a direct port of [ESRGAN by xinntao](https://github.com/xinntao/ESRGAN), so all results, accomplishments, and such that ESRGAN does, VSGAN will do too.

Using the right pre-trained model, on the right image, can have tremendous results.  
Here's an example from a US Region 1 (NTSC) DVD of American Dad running with VSGAN (model not public)
![Example 1](examples/cmp_1.png)

## :camera: Qualitive Comparisons against other Super-Resolution Strategies

Following comparisons were taken from [ESRGAN's repo](https://github.com/xinntao/ESRGAN)
![qualitive1](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_01.jpg)
![qualitive2](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_02.jpg)
![qualitive3](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_03.jpg)
![qualitive4](https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_04.jpg)

## :wrench: Installation and Usage
[Check out the Wiki](https://github.com/rlaPHOENiX/VSGAN/wiki), it will explain everything you may need to know.
