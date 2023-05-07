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
        load(r"C:\Users\PHOENiX\Documents\PSNR_x4_DB.pth").\
        apply().\
        load(r"C:\Users\PHOENiX\Documents\4X_DoubleRunExample.pth").\
        apply(overlap=16).\
        apply(overlap=32).\
        clip

For more information see (:ref:`Getting Started <getting started>`).

Features of VSGAN
-----------------

.. _VapourSynth: https://vapoursynth.com/doc/installation.html
.. _NLE: https://en.wikipedia.org/wiki/Non-linear_editing

- VapourSynth_ — Transform, Filter, or Enhance your input video, or the VSGAN result with VapourSynth,
  a Script-based NLE_.
- :ref:`Easy Model Chaining <Chaining calls, or Models>` — You can chain models or re-run the model twice-over
  (or more).
- :ref:`Seamless Tiling` — Have low VRAM? Don't worry! The Network will be applied in quadrants of the
  image to reduce up-front VRAM usage.
- Supports All RGB formats — You can use *any* RGB video input, including float32 (e.g., RGBS) inputs.
- No Frame Extraction Necessary — Using VapourSynth you can pass a Video directly to VSGAN, without any frame
  extraction needed.
- Repeatable Edits — Any edit you make in the VapourSynth script with or without VSGAN can be re-used for any
  other video.
- :ref:`Freedom <License>` — VSGAN is released under the MIT License, ensuring it will stay free, with the
  ability to be used commercially.

Supported Models
----------------

`ESRGAN <https://arxiv.org/abs/1809.00219>`_
  Enhanced Super-Resolution Generative Adversarial Networks.

`ESRGAN+ <https://arxiv.org/abs/2001.08073>`_
  Further Improving Enhanced Super-Resolution Generative Adversarial Network.

`Real-ESRGAN <https://arxiv.org/abs/2107.10833>`_
  Training Real-World Blind Super-Resolution with Pure Synthetic Data.

`A-ESRGAN <https://arxiv.org/abs/2112.10046>`_
  Training Real-World Blind Super-Resolution with Attention U-Net Discriminators.

`EGVSR <https://arxiv.org/abs/2107.05307>`_
  Real-Time Super-Resolution System of 4K-Video Based on Deep Learning.

`SwinIR <https://arxiv.org/abs/2108.10257>`_
  Image Restoration Using Swin Transformer.

`HAT <https://arxiv.org/abs/2205.04437>`_
  Activating More Pixels in Image Super-Resolution Transformer.

Quick shoutout to pvsfunc.PD2V
------------------------------

A lot of Super-Resolution users work with old low-resolution media.
If you plan to work with DVD video files, or generally NTSC/PAL standard MPEG-1/2 media, you should take a look
at my other project `pvsfunc <https://github.com/rlaphoenix/pvsfunc>`_.

In pvsfunc there's a class `PD2V` which is intended specifically for DVD video files, but can work on other
sourced MPEG-1/2 media. It optimally loads the video data Frame-accurately with various helper functions.

* Frame-accurate frame-serving. Check for and supports mixed scan-type inputs.
* `recover()` — Recover progressive frames in an interlaced stream in some scenarios, usually on animation or fake-interlacing.
* `floor()` or `ceil()` — Convert Variable Frame-rate (VFR) to Constant Frame-rate. `floor()` is experimental.
* `deinterlace()` — Deinterlace efficiently, only when needed. This is only a wrapper, an actual deinterlace function is still
  needed.
* All of this and more with self-chaining support, just like VSGAN.

Example Results
---------------

Mickey's Christmas Carol
^^^^^^^^^^^^^^^^^^^^^^^^

.. vimeo:: 657905289
   :aspect: 4:3
   :width: 100%

`This is what the official Disney Blu-ray looks like <https://vimeo.com/115284525>`_...

American Dad S01E01
^^^^^^^^^^^^^^^^^^^

.. vimeo:: 382251167
   :aspect: 3:2
   :width: 100%

.. thumbnail:: _static/images/examples/american_dad_s01e01/before.webp
   :group: ex1
   :width: 49%

   Original Input (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/american_dad_s01e01/after.webp
   :group: ex1
   :width: 49%

   Private 4x Model Applied (2880x1920 -> 1620x1080)

This model was trained to fix inaccuracies in the DVD's color, remove Halo'ing/Glow, and remove Chroma Droop. The
result is a very crisp output for a show originally animated in SD.

Family Guy S01E01
^^^^^^^^^^^^^^^^^

.. thumbnail:: _static/images/examples/family_guy_s01e01/before.webp
   :group: ex2
   :width: 49%

   Original Input (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/family_guy_s01e01/after.webp
   :group: ex2
   :width: 49%

   Private 4x Model Applied (2880x1920 -> 1620x1080)

This model was trained to fix inaccuracies in the DVD's color, remove Halo'ing/Glow, and remove Chroma Droop. The
result is a very crisp output for a show originally animated in SD. Do note that the warping/stretch on the edges
is an animation/dvd edit and not caused by VSGAN or the model.

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
