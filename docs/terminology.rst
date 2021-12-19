Terminology
===========

Single Image
    Using the data from one image for one output image.

Model
    The Generator Network of an Architecture.

Network
    Module itself that comprises of other modules (layers) that perform operations on data.

Architecture
    A combination of Networks with specific purposes. E.g., a GAN network would consist of a Generator (G)
    Network as well as a Discriminator (D) Network.

Generator (G) Network
    Transformed input data to new output data based on the Networks layers.

Discriminator (D) Network
    Essentially tries to tell if a Networks output is fake/bad. Think of it as a human quickly comparing
    the G Network's output to the original GT image to see if it's a good result.

    This network would only be used for Training purposes, and generally wouldn't be used by VSGAN.

Super-Resolution (SR)
    Result of a model with a > 1x scale output. Aka, Upscaling, Upconverting, Resizing.

Generative Adversarial Network (GAN)
    Adversarial which a Generator (G) network generates data, and a Discriminator (D) tries to detect if the
    generated image is perceived as fake.

Low-Resolution (LR)
    The low-resolution input image/data. The data you wish to transform with the model.

Ground Truth (GT) or High-Resolution (HR)
    The original high resolution image/data. This data would be used for your Discriminator while training,
    or for comparison.
