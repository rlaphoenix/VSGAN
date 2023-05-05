Terminology
===========

Single Image
    Using the data from one image for one output image.

Model
    A file containing the learned parameters of a Network that can be used for inference or further training.

Architecture
    One or more interconnected Networks. For example, a GAN architecture would consist of both a Generator (G)
    Network and a Discriminator (D) Network.

Network
    A collection of interconnected components or nodes that work together to learn patterns in data.

Block
    A function or set of mathematical operations that processes input data in a specific way, such as convolution,
    normalization, or padding.

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
