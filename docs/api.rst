Interface
=========

This part of the documentation covers all the interfaces of VSGAN.

The initial clip provided to the :class:`VSGAN <VSGAN>` object will be the base clip used by all calls.
Each time you run a model, it will apply to the base clip, and then overwrite it.

Once you have done all the calls you wish to do on the clip, get the final clip by taking the `clip`
property of the :class:`VSGAN <VSGAN>` object.

.. autoclass:: vsgan.VSGAN
   :member-order: bysource
   :members:
