Models
======

Model files are PyTorch ``.pth`` files with trained network parameters stored within. They are the same as
`state dictionaries`. Think of it as configuration of a Network based on the training it's had.

VSGAN must support the model file's specific architecture for the model file to work. A list of
:ref:`Supported Models` are listed on the homepage.

Where can I get Model files?
----------------------------

- `Upscale.wiki Model Database <https://upscale.wiki/wiki/Model_Database>`_
- `Game Upscale Discord <https://discord.gg/cpAUpDK>`_

Know any more places to find trained models? Let me know!

Do I need to train a Model file?
--------------------------------

Yes and no. VSGAN is not an ESRGAN model trainer, it's a model tester. You will need to provide your own
trained model file for VSGAN to use. If you do not have a Model file, you can either find one trained by
someone else, or train your own.

How can I train my own Model file?
----------------------------------

You may wish to train your own model file for various reasons. E.g., the model files you find don't work
for your scenario, or you feel like you could get a better result.

There's a few things you must realise about training a model.

1. It takes a lot of patience, and constant running of your GPU hardware at high-usage.
2. It takes a lot of effort to create the dataset that will be used to train the model how you like.
3. A lot of storage may be used for the dataset, and using an SSD (NVMe if possible) is recommended.
4. Writing a lot to an SSD (or NVMe) can be dangerous to it's lifespan if you constantly write a lot of
   data.
5. Among all of this, there's quite a lot more to learn than what you would learn to use VSGAN.
   There will be a lot of words like Losses, LR, HR, GT, Validation, Act, Norm, Layer, Arch, and so on.

If you still feel like you want to give it a try, go for it! Check out `traiNNer <https://github.com/victorca25/traiNNer>`
it's a PyTorch training and testing toolbox for various Model Architectures. For instructions on using
traiNNer, check their readme. VSGAN will not provide help for training.
