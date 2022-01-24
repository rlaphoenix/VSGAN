Installation
============

This part of the documentation covers the installation of VSGAN.
The first step to using any software package is getting it properly installed.

Hardware Requirements
---------------------

* **GPU** supporting CUDA 9.2 or newer. A `CUDA Compute Capability score <https://developer.nvidia.com/cuda-gpus#compute>`_
  of 6 or higher is recommended. A score less than or equal to 2 may not work at all.
* **CPU** doesn't really matter. But you need one. It won't be used too much but make sure it's not going to bottle-neck
  your GPU.

**No supported GPU?** You *can* use your CPU but it really is not recommended. It may lower the life-span quite
quickly and will get very very hot. Your PC will also slow to a *crawl* with your mouse barely functioning. I will
not be responsible for whatever you choose to do.

Software Requirements
---------------------

Install these in the order listed. The latest version of all dependencies are recommended where possible.
Do note that VapourSynth and PyTorch may not always support the latest version of Python, in this case you
may need to install an older version to continue.

.. _Python: https://python.org
.. _VapourSynth: https://vapoursynth.com/doc/installation.html
.. _PyTorch: https://pytorch.org/get-started/locally

1. Python_ 3.7 or newer. Must be a version supported by both VapourSynth_ and PyTorch_.
2. VapourSynth_ r48 or newer.

   - ⚠️ The Pip/PyPI package `VapourSynth` is *not* what you want to install.
3. PyTorch_ 1.6.0 or newer.

   - If you want to use your GPU, choose the latest CUDA version when installing.
   - If you want CUDA installed system-wide, choose `CPU` instead of `CUDA` and then
     install CUDA manually. 

Installing VSGAN
----------------

.. code-block:: shell

  $ python -m pip install --user VSGAN

.. warning::

  Portable installations should use the `python` binary in their portable directory by full path instead
  of `python` in CLI calls. For ``pip ...`` calls, do ``path/to/portable/python -m pip ...`` instead.

.. note::

  You may also install from source; Check out :ref:`Building`.

Updating VSGAN
--------------

For PIP/PyPI installations it's as simple as the ``-U``.

.. code-block:: shell

    $ python -m pip install -U VSGAN
    $ python -m pip install --user --force VSGAN==1.2.3  # force a specific version

If you are trying to update the installed Source Code, just re-do the last call of
the original installation steps. You may want to ignore checking dependencies for
updates using ``--no-deps``, or to force re-installation with ``--force``.

.. code-block:: shell

    $ python -m pip install --user .
