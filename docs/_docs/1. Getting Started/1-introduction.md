---
title: "Introduction"
permalink: /
excerpt: "Documentation and Introduction to VSGAN, ESRGAN, and VapourSynth."
last_modified_at: 2021-01-16T11:58:00-00:00
toc: true
classes: wide
---

VSGAN is a Single Image Super-Resolution Generative Adversarial Network (GAN) which uses the VapourSynth processing framework to handle input and output image data.

**Note:** The GAN Architecture is exactly that of [ESRGAN](https://github.com/xinntao/ESRGAN) by [xinntao](https://github.com/xinntao). All accomplishments of ESRGAN will also be achieved with VSGAN.
{: .notice--success}

**Warning:** Depending on the use-case, performance may not match the original ESRGAN, however, it won't be drastically slower.
{: .notice--warning}

## Quick Terminology Gloss-over

| Term                                 | Meaning                                                                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Single Image                         | Using the data from one image for one output image.                                                                                            |
| Super-Resolution (SR)                | Known otherwise as Upscaling/Upconverting/Resizing.                                                                                            |
| Generative Adversarial Network (GAN) | Adversarial which a Generator (G) network generates data, and a Discriminator (D) tries to detect if the generated image is perceived as fake. |
| Ground Truth (GT)                    | The original high resolution image/data. Also known as HR (High-resolution).                                                                   |

## Introduction to ESRGAN

ESRGAN: Enhanced super-resolution generative adversarial network. First place winner in [PIRM2018-SR competition](https://www.pirm2018.org/PIRM-SR.html) (Region 3) with the best perceptual index.
The paper is accepted to [ECCV2018 PIRM Workshop](https://pirm2018.org/).

It's an improvement of [SRGAN](https://arxiv.org/abs/1609.04802) in three aspects:

1. Adopt a deeper model using Residual-in-Residual Dense Block (RRDB) without batch normalization layers.
2. Employ [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) instead of the vanilla GAN.
3. Improve the perceptual loss by using the features before activation.

> Authors: Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., & Loy, C. (2018)

### Comparisons

<figure>
   <a href="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_01.jpg">
      <img src="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_01.jpg" alt="Baboon Super-resolution Architecture Comparison 1">
   </a>
   <a href="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_02.jpg">
      <img src="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_02.jpg" alt="Face Super-resolution Architecture Comparison 2">
   </a>
   <a href="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_03.jpg">
      <img src="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_03.jpg" alt="Castle Super-resolution Architecture Comparison 3">
   </a>
   <a href="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_04.jpg">
      <img src="https://raw.githubusercontent.com/xinntao/ESRGAN/master/figures/qualitative_cmp_04.jpg" alt="Wheat Super-resolution Architecture Comparison 4">
   </a>
   <figcaption>Super-resolution Architectures Comparison between Bicubic (Algorithm), SRCNN, EDSR, RCAN, EnhanceNet, SRGAN, and ESRGAN. Source: ESRGAN</figcaption>
</figure>

### Training Results

|                          Method                          | Training dataset |       Set5       |       Set14      |      BSD100      |     Urban100     |     Manga109     |
| :------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) |        291       |   30.48/0.8628   |   27.50/0.7513   |   26.90/0.7101   |   24.52/0.7221   |   27.58/0.8555   |
|    [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch)   |       DIV2K      |   32.46/0.8968   |   28.80/0.7876   |   27.71/0.7420   |   26.64/0.8033   |   31.02/0.9148   |
|        [RCAN](https://github.com/yulunzhang/RCAN)        |       DIV2K      |   32.63/0.9002   |   28.87/0.7889   |   27.77/0.7436   |   26.82/ 0.8087  |   31.22/ 0.9173  |
|                       RRDB (ESRGAN)                      |       DF2K       | **32.73/0.9011** | **28.99/0.7917** | **27.85/0.7455** | **27.03/0.8153** | **31.66/0.9196** |

<figcaption>
   The RRDB PSNR oriented model trained with DF2K dataset (<a href="https://data.vision.ee.ethz.ch/cvl/DIV2K">DIV2K</a> & <a href="http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">Flickr2K</a>,
   proposed in <a href="https://github.com/LimBee/NTIRE2017">EDSR</a>) compared against other Super-resolution Architectures. Best result in each validation marked bold. Source: ESRGAN
</figcaption>

### More information

If you wish to learn more about training, stats, or learn it's issues, then check out the [ESRGAN README](https://github.com/xinntao/ESRGAN/blob/master/README.md).

## Benefits of VSGAN over ESRGAN

Since VapourSynth is primed to pass a frame sequence to a script; this could be considered a Video Super-Resolution Network that doesn't take advantage of neighbouring frame data. This allows you to pass hundreds of thousands of frames from video files through ESRGAN from an input video file, removing the need to extract the frames out of a video which takes a lot of time, processing power, and file space.

VapourSynth provides an extreme amount of image processing functionality for pretty much anything you can imagine to do with an image programmatically. For example, pre and post filtering of the frames with scaling, cropping, flipping, rotating, denoising, color augmentations, comparisons, and more. Not to mention advanced possibilities like fixing the Chroma location positioning (aka chroma bleed) or chroma size (chroma droop), Chroma-channel noise, Dot-crawl, Rainbowing, and more!

You can see a database of readily-available scripts and plugins for all kinds of purposes on [VSDB](https://vsdb.top). For help with VapourSynth head to [Doom9](https://forum.doom9.org) or [VideoHelp](https://forum.videohelp.com).
{: .notice--info}

### Example VSGAN output

<figure>
   <a href="https://raw.githubusercontent.com/rlaPHOENiX/VSGAN/master/examples/cmp_1.png">
      <img src="https://raw.githubusercontent.com/rlaPHOENiX/VSGAN/master/examples/cmp_1.png" alt="">
   </a>
   <figcaption>
      Before (left): 720x480 resolution frame from American Dad S01E01 (USA R1 NTSC DVD). After (right): A private ESRGAN 4x scale model
      using VSGAN on the before frame. This model was trained to fix inaccuracies in the DVD's color, remove Halo'ing/Glow,
      and remove Chroma Droop. The result is a very crisp output for a show originally animated in SD. It's using VapourSynth's
      core.std.StackHorizontal to provide a before and after comparison, with core.text.Text for the Before and After labels.
   </figcaption>
</figure>
