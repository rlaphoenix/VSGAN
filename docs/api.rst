Interface
=========

This part of the documentation covers all the interfaces of VSGAN.

The initial clip provided to the Network will be the base clip used by all further calls.
Each time you run a model, it will apply to the base clip, and then overwrite it.

Once you have done all the calls you wish to do on the clip, get the final clip by taking the `clip`
property of the Network object.

Networks
--------

.. autoclass:: vsgan.archs.ESRGAN
   :member-order: bysource
   :members:

.. autoclass:: vsgan.archs.EGVSR
   :member-order: bysource
   :members:

Utilities
---------

.. automodule:: vsgan.utilities
    :members:
