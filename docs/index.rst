VSGAN
==================================================

Release v\ |version|. (:ref:`Installation <installation>`)

.. image:: https://pepy.tech/badge/vsgan
    :target: https://pepy.tech/project/vsgan

.. image:: https://img.shields.io/pypi/l/vsgan.svg
    :target: https://pypi.org/project/vsgan

.. image:: https://img.shields.io/pypi/wheel/vsgan.svg
    :target: https://pypi.org/project/vsgan

.. image:: https://img.shields.io/pypi/pyversions/vsgan.svg
    :target: https://pypi.org/project/vsgan

PyTorch-based Super-Resolution and Restoration Image Processing Module for VapourSynth.

-------------------

**Short example**:

.. code:: shell

    from vsgan import ESRGAN
    clip = ESRGAN(clip, device="cuda").\
        load(r"C:\Users\PHOENiX\Documents\Denoise_x1.pth").\
        apply().\
        load(r"C:\Users\PHOENiX\Documents\PSNR_x2.pth").\
        apply(overlap=16).\
        apply(overlap=32).\
        clip

This runs an ESRGAN denoising model at x1 scale on the input clip using the Nvidia CUDA GPU as the device.
It then loads a new PSNR-optimized model at 2x scale to the same GPU device and applies it twice over.
Finally, it returns the internal clip that had the models applied to it.

Note that by chaining the application of the 2x scale model twice-over, the resulting clip is now four times
the size of the original clip. Normally, this is a very expensive operation. However, with VSGAN, this is
possible thanks to its automatic tiling system.

For more information, see (:ref:`Getting Started <getting started>`).

Features of VSGAN
-----------------

.. _VapourSynth: https://vapoursynth.com/doc/installation.html
.. _NLE: https://en.wikipedia.org/wiki/Non-linear_editing

- VapourSynth_ — Transform, Filter, or Enhance your input video, or the VSGAN result with VapourSynth,
  a Script-based NLE_.
- :ref:`Model Chaining <Chaining calls, or Models>` — Apply multiple models on the input video, or apply models
  multiple times over.
- :ref:`Seamless Tiling` — Reduce up-front memory usage by automatically applying the model on quadrants of the
  input image at a time, without leaving any noticeable seams.
- Supports All RGB formats — You can use any RGB video input of any bit depth.
- No Frame Extraction Necessary — Using VapourSynth you can pass a Video directly to VSGAN, without any frame
  extraction needed.
- Repeatable Edits — Any edit you make in the VapourSynth script with or without VSGAN can be re-used for any
  other video.
- :ref:`Freedom <License>` — VSGAN is released under the MIT License, ensuring it will stay free, with the
  ability to be used commercially.

Supported Model Architectures
-----------------------------

`ESRGAN <https://arxiv.org/abs/1809.00219>`_
  Enhanced Super-Resolution Generative Adversarial Networks.

  `ESRGAN+ <https://arxiv.org/abs/2001.08073>`_
    Further Improving Enhanced Super-Resolution Generative Adversarial Network.

  Also supports other models with only differences during training, i.e., blind degradation models like
  A-ESRGAN and BSRGAN.

`Real-ESRGAN <https://arxiv.org/abs/2107.10833>`_
  Training Real-World Blind Super-Resolution with Pure Synthetic Data.

`EGVSR <https://arxiv.org/abs/2107.05307>`_
  Real-Time Super-Resolution System of 4K-Video Based on Deep Learning.

`SwinIR <https://arxiv.org/abs/2108.10257>`_
  Image Restoration Using Swin Transformer.

`HAT <https://arxiv.org/abs/2205.04437>`_
  Activating More Pixels in Image Super-Resolution Transformer.

Quick shoutout to MPGG
----------------------

.. _deinterlacing: https://en.wikipedia.org/wiki/Deinterlacing
.. _detelecining: https://en.wikipedia.org/wiki/Telecine#Reverse_telecine_(a.k.a._inverse_telecine_(IVTC),_reverse_pulldown)

If you plan to work with DVD-Video files, or general MPEG-1/2 media, you should take a look at my other project
`MPGG <https://github.com/rlaphoenix/mpgg>`_: a Streamlined MPEG-1 and MPEG-2 source loader and helper utility for
VapourSynth.

I always recommend deinterlacing_ and detelecining_ sources before using VSGAN, and MPGG will greatly assist in doing
so, as well as a lot more! It is designed specifically for DVD-Video files, but can work on other MPEG-1/2 media, e.g.,
Camcorders, VHS digitizers, old console capture cards, and more.

Example Results
---------------

Mickey's Christmas Carol
^^^^^^^^^^^^^^^^^^^^^^^^

.. vimeo:: 657905289
   :aspect: 10:7
   :width: 100%

.. raw:: html

   <br>

.. thumbnail:: _static/images/examples/mickeys_christmas_carol/dvd_before.webp
   :group: ex1_dvd
   :width: 49%

   Before (Unaltered EUR PAL R2 DVD 720x576, DAR of 768x576)

.. thumbnail:: _static/images/examples/mickeys_christmas_carol/dvd_after.webp
   :group: ex1_dvd
   :width: 49%

   After (2X_DigitalFilmV5_Lite with some pre-and-post-filtering)

There wasn't really any issues with the original source to fix, but it did improve the sharpness while not over
doing it or causing other artifacts, unlike the official Disney Blu-ray, which did a terrible job.

   Comparison against the official Disney Blu-ray:

   .. thumbnail:: _static/images/examples/mickeys_christmas_carol/bd_disney.webp
      :group: ex1_bd
      :width: 49%

      Before (Original Blu-ray release, completely unaltered)

   .. thumbnail:: _static/images/examples/mickeys_christmas_carol/bd_rlaphoenix.webp
      :group: ex1_bd
      :width: 49%

      After (2X_DigitalFilmV5_Lite with some pre-and-post-filtering, cropped to the same aspect ratio as Disney's Blu-ray)

   As you can see they over-sharpened, posterized the image, removed detail, caused Halo'ing, added a sort of green
   tint across the picture, and cropped in to 16:9 from a 4:3 source. Yikes!

   I did not mix up the labels of these images. The one labeled as the Official Blu-ray release, really is like that!

`(VapourSynth Script) <https://gist.github.com/rlaphoenix/0957656a97559c397ec12544743f2898>`_

American Dad S01E01
^^^^^^^^^^^^^^^^^^^

.. vimeo:: 382251167
   :aspect: 8:5
   :width: 100%

.. raw:: html

   <br>

.. thumbnail:: _static/images/examples/american_dad_s01e01/before.webp
   :group: ex2
   :width: 49%

   Original Input (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/american_dad_s01e01/after.webp
   :group: ex2
   :width: 49%

   Private 4x Model Applied (2880x1920 -> 1620x1080)

Fixed Halo'ing/Glow around outlines, and a bit of Chroma Droop. However, the colors between images in the dataset
pair had a mistake so it learned to modify the colors incorrectly.

Family Guy S01E01
^^^^^^^^^^^^^^^^^

.. thumbnail:: _static/images/examples/family_guy_s01e01/before.webp
   :group: ex3
   :width: 49%

   Original Input (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/family_guy_s01e01/after.webp
   :group: ex3
   :width: 49%

   Private 4x Model Applied (2880x1920 -> 1620x1080)

Fixed slight inaccuracies in the Color, Halo'ing/Glow around outlines, Chroma Location Misplacement, as well as a
bit of Chroma Droop (on some scenes only). This is an almost perfect result (in my opinion). Do note that the
warping/stretch on the edges is an animation/dvd edit and not caused by VSGAN or the model.

.. toctree::
   :hidden:

   installation
   building
   getting-started
   models
   api
   troubleshooting
   terminology
   changelog
   license

.. toctree::
   :caption: Resources
   :hidden:

   VapourSynth Docs <https://vapoursynth.com/doc>
   Doom9 <https://forum.doom9.org>
   VideoHelp <https://forum.videohelp.com>

.. toctree::
   :caption: Project
   :hidden:

   Source <https://github.com/rlaphoenix/VSGAN>
   Issues <https://github.com/rlaphoenix/VSGAN/issues>
   Discussions <https://github.com/rlaphoenix/VSGAN/discussions>
