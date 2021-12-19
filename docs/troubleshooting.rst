Troubleshooting
===============

#1 — I'm using a portable installation strategy and it cannot install and/or locate VSGAN.
--------------------------------------------------------------------------------------------

The usual reason for this is you mistakenly installed some or all of the dependencies of VSGAN to your
system's Python installation and not the portable Python. This is because ``python`` and ``pip`` in your
CMD/Terminal will refer to the system installation rather than the portable python's binaries. You must
directly refer to the portable binaries instead e.g., ``path\to\portable\python.exe``.

To call pip from a portable installation, do e.g., ``path\to\portable\python.exe -m pip install vsgan``.

Once you rectify this mistake, ensure you re-do all the installation steps related to python and pip,
like PyTorch. You may also want to check your system installation to see if you have them on your system
installation which you may not want. Run ``pip freeze`` and it will list all installed packages, which you
can then uninstall.

#2 — Either NVIDIA CUDA or the device (cuda) isn't available.
---------------------------------------------------------------

This will happen when you do not have NVIDIA CUDA installed, nor have you chosen to install PyTorch with
CUDA vendored-in.
You most likely went to the PyTorch Getting Started page and forgot to choose "Pip", "Python", "CUDA X.X",
and instead chose "CPU".

However, the less likely answer is that your GPU may not support the minimum CUDA version, or was not found
by NVIDIA CUDA or PyTorch.

#3 — error in LoadLibraryA
----------------------------

This indicates an error or incompatability with your PyTorch installation. This could happen if you have no
CUDA-capable GPU yet installed PyTorch with CUDA instead of the CPU-only option. There's also the
possibility of the error happening the other way around too.

It may also be a good idea to ensure your GPU drivers are up-to-date. You can check using NVIDIA's GeForce
Experience software.

#4 — Error(s) in loading state_dict, Missing key(s) in state_dict
-------------------------------------------------------------------

You may be on an old version of VSGAN that does not support this kind of ESRGAN model, or it may be a
corrupt file.
