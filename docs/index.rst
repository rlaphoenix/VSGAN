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

**VSGAN** Single Image Super-Resolution Generative Adversarial Network (GAN)
which uses the VapourSynth processing framework to handle input and output
image data.

-------------------

**Short example**::

   clip = VSGAN(clip, device="cuda").\
     load_model(r"C:\Users\PHOENiX\Documents\PSNR_x4_DB.pth").\
     run().\
     load_model(r"C:\Users\PHOENiX\Documents\4X_DoubleRunExample.pth").\
     run(overlap=16).\
     run(overlap=32).\
     clip

For more information see (:ref:`Usage <usage>`).

Features of VSGAN
-----------------

- Script-based NLE — Transform, Filter, or Enhance your input video, or the Model's result.
- Easy Model Chaining — You can chain models or re-run the model twice-over (or more) very easily with VSGAN.
- Supports All RGB formats — You can use *any* RGB video input, including float32 (e.g., RGBS) inputs.
- Seamless Chunking support — Have low VRAM? Don't worry! Use `overlap` argument to chunk seamlessly to lower
  VRAM requirements.
- No Frame Extraction Necessary — Using VapourSynth you can pass a Video directly to VSGAN, without any frame
  extraction necessary.
- Repeatable Edits — Any edit you make in the VapourSynth script with or without VSGAN can be re-used for any
  other video.
- Freedom — VSGAN is released under the MIT License, ensuring it will stay free, with the ability to be used
  commercially.

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
   :alt: American Dad S01E01 (USA NTSC R1 DVD 720x480)

   American Dad S01E01 (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/american_dad_s01e01/after.webp
   :group: ex1
   :width: 49%
   :alt: American Dad S01E01 with Private Model Applied

   American Dad S01E01 with Private Model Applied

This model was trained to fix inaccuracies in the DVD's color, remove Halo'ing/Glow, and remove Chroma Droop. The
result is a very crisp output for a show originally animated in SD.

Family Guy S01E01
^^^^^^^^^^^^^^^^^

.. thumbnail:: _static/images/examples/family_guy_s01e01/before.webp
   :group: ex2
   :width: 49%
   :alt: Family Guy S01E01 (USA NTSC R1 DVD 720x480)

   Family Guy S01E01 (USA NTSC R1 DVD 720x480)

.. thumbnail:: _static/images/examples/family_guy_s01e01/after.webp
   :group: ex2
   :width: 49%
   :alt: Family Guy S01E01 with Private Model Applied

   Family Guy S01E01 with Private Model Applied

This model was trained to fix inaccuracies in the DVD's color, remove Halo'ing/Glow, and remove Chroma Droop. The
result is a very crisp output for a show originally animated in SD. Do note that the warping/stretch on the edges
is an animation/dvd edit and not caused by VSGAN or the model.

Supported Models
----------------

The accomplishments of each model will be reflected equally when used with VSGAN. The only difference will be
the API as to which you use that model, which is now via VSGAN calls.
All models unless explicitly stated, supports only models of integer scale. i.e., scales in a power of 2.

`ESRGAN <https://github.com/xinntao/ESRGAN>`_
  Enhanced Super-Resolution Generative Adversarial Networks. Supports both old and new-arch models of any scale.

Pages
-----

.. toctree::
   :maxdepth: 2

   installation
   usage
   api
   troubleshooting
   terminology
   license
