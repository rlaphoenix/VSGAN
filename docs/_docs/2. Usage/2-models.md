---
title: "Models"
permalink: /models/
excerpt: "Settings for configuring and customizing the theme."
last_modified_at: 2021-01-16T11:33:00-00:00
toc: false
classes: wide
---

A PyTorch model file (.pth) is used to initialize the ESRGAN model's parameters to what it was when training finished or ended. These parameters influence the output.

VSGAN is not an ESRGAN model trainer, it's a model tester. You will need to provide your own model file for VSGAN to use.
{: .notice--warning}

ESRGAN model files have two different state dictionary versions, the older (aka [old-arch](https://github.com/xinntao/BasicSR/releases/tag/v0.0)) version, and the
newer/current (aka [new-arch](https://github.com/xinntao/BasicSR/releases/tag/v1.0.0)) version. The only difference is the naming of the state dict keys.
VSGAN automatically supports both versions by renaming new-arch model keys as old-arch keys, detects model scales, and supports any scale.
{: .notice--info}

**Need a model file?** You can find models on the [Game Upscale Discord](https://discord.gg/cpAUpDK) or their [Upscale.wiki Model Database](https://upscale.wiki/wiki/Model_Database).
{: .notice--info}

**Want to train your own?** See [BasicSR by xinntao](https://github.com/xinntao/BasicSR) which is a model trainer with support for ESRGAN and other architectures.
BasicSR trains the previously explained new-arch models (since after commit [9bbc011](https://github.com/xinntao/BasicSR/releases/tag/v0.0)), and as stated there's no
differences on the model file other than the naming of keys. However, for it's training code it *does* have a difference, it restricts model training scale to 4x.
If you want to use a different scale, or simply want to train old-arch models, you can use the popular [Victorca25](https://github.com/victorca25/BasicSR) fork.
{: .notice--info}