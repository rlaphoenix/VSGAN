===============
Getting started
===============

Welcome! This tutorial highlights VSGAN's core features; for further details,
see the links within, or the documentation index which has links to conceptual
and API doc sections.

Importing Networks and Applying Models
======================================

As of v1.6.0, VSGAN supports more than just ESRGAN, and so to execute models you
must now import the specific Network class for the model.

For example, to apply a ESRGAN model:

.. code:: shell

    >>> import vapoursynth as vs
    >>> from vapoursynth import core
    >>> from vsgan import ESRGAN
    >>> clip = core.std.BlankClip(width=720, height=480, format=vs.RGBS)
    'VideoNode
        Format: RGBS
        Width: 720
        Height: 480
        Num Frames: 240
        FPS: 24'
    >>> esrgan = ESRGAN(clip, device="cuda")
    <vsgan.networks.esrgan.ESRGAN object>
    >>> esrgan.load(r'C:/Users/John/Documents/PSNR_x4_DB.pth')
    <vsgan.networks.esrgan.ESRGAN object>
    >>> esrgan.apply()
    <vsgan.networks.esrgan.ESRGAN object>
    >>> clip = esrgan.clip
    'VideoNode
        Format: RGBS
        Width: 2880
        Height: 1920
        Num Frames: 240
        FPS: 24'

A list of :ref:`Supported Models` are listed on the homepage, and a list of Networks are
listed in the :ref:`Interface` documentation.

You have the choice to load any new model at any point. You can even apply the model
as many times as you wish, with any settings you wish.

The Network classes represent a Video Clip (:class:`vs.VideoNode`) that provides
Model APIs, such as ``apply()`` which applies the model on the clip's frames.

.. warning::
    The input clip must be an RGB clip. RGBS is recommended for F32 RGB input.
    You can convert to RGB using any of VapourSynth's
    `core.resize.* <https://vapoursynth.com/doc/functions/video/resize.html>`_
    functions. Example `core.resize.Spline16(clip, format=vs.RGBS)`. Do note that
    the resize algorithm will affect chroma conversion.

.. note::
    Don't forget to take the final clip from the Network object by taking its
    `.clip` property.

Chaining calls, or Models
=========================

All functions of the Network classes return itself, this allows chaining of
commands, like chaining model runs by simply appending another function call.

.. code:: shell

    >>> from vsgan import ESRGAN
    >>> clip = core.std.BlankClip(width=720, height=480, format=vs.RGBS)
    >>> clip = ESRGAN(clip, device="cuda").\
            load(r'C:/Users/John/Documents/PSNR_x4_DB.pth').\
            apply().\
            clip  # get final clip
    'VideoNode
        Format: RGBS
        Width: 2880
        Height: 1920
        Num Frames: 240
        FPS: 24'

This allows you to easily chain multiple models or even multiple runs of a model
in one swift call:

.. code:: shell

    >>> from vsgan import ESRGAN
    >>> clip = core.std.BlankClip(width=720, height=480, format=vs.RGBS)
    >>> clip = ESRGAN(clip, device="cuda").\
            load(r'C:/Users/John/Documents/1x_Unresize.pth').\
            apply().\
            apply().\  # run twice
            load(r'C:/Users/John/Documents/RealESRGAN_x4plus.pth').\
            apply().\  # change model and run once
            clip
    'VideoNode
        Format: RGBS
        Width: 2880
        Height: 1920
        Num Frames: 240
        FPS: 24'

Seamless Tiling
===============

*new in v1.4.0, reworked in v1.6.0*

Tiling is used to chunk the clip into 4 quadrants. The model is then operated on each
quadrant separately. This effectively cuts VRAM requirements by up to 75%.

Networks now support recursive tiling which is now automatically attempted if you have
run out of VRAM. It will automatically attempt to apply the model using as little tiling
as possible, perhaps even no tiling if it can.

The overlap value defines how much each tile will overlap with data from its neighbouring
tiles. This is to remove seam artefacts common with most Networks.

.. note::
    Unlike other chunking solutions, this one does not leave a seam or
    edge-artifacts as long as the `overlap` parameter is a high enough value
    for the resolution.

Finding a good overlap value
----------------------------

Here's some examples that show what the seam looks like, with various overlap
values. Notice the striking edge-artifacts down the center of both axes.
These are actually artifacts cased by the model on the edges of all 4 quadrants.

.. thumbnail:: _static/images/seams/clearly-visible.webp
    :group: seam
    :width: 49%
    :title: Overlap of 1 with very noticeable seams.

.. thumbnail:: _static/images/seams/barely-noticeable.webp
    :group: seam
    :width: 49%
    :title: Overlap of 10 with barely noticeable seams.

The overlap value aims to mask these naturally by extending the image input past
the boundaries of each quadrant. Essentially making each quadrant slightly larger
than it should be. This makes the model cause the artifacts on the edge of the
picture that will be trimmed away when re-merging as one singular picture.

The first image used ``overlap=1`` and is an example of an overlap amount that
isn't enough. The second image had an overlap amount that is still just slightly
too small. You can just barely make out some seams. In cases like this, the
overlap amount should be further enlarged. A good overlap value should result
in no seam being noticeable at all. Not even a spec of it.

.. note::
    The value of overlap should generally be a multiple of ``16``. The larger
    the input resolution, the larger overlap may need to be set. Avoid using
    a value excessively large, but ensure the value is enough to be rid of the
    seam completely on all scene types (bright, dark, and so on).

.. note::
    The larger the overlap value, the more VRAM you will use per-quadrant,
    and the slower it may perform. Regardless, the amount of VRAM you save
    just by using chunk-mode at all should be worth it.
