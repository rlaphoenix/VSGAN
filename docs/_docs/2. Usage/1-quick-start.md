---
title: "Quick Start"
permalink: /quick-start/
excerpt: "Settings for configuring and customizing the theme."
last_modified_at: 2021-01-16T13:08:00-00:00
toc: false
classes: wide
---

Here's a quick example of VSGAN usage. Further Information on the classes and functions used can be found in the [Functions]({{ '/vsgan-class/' | absolute_url }}) documentation.

```py
import vapoursynth as vs

from vsgan import VSGAN

# ...

# PyTorch device, e.g. "cpu", "cuda", "cuda:0", "cuda:1", 0, 1, ..., e.t.c
device = "cuda"

# ESRGAN model file, see {{ '/models/' | absolute_url }}
# tip: prepend path with r" ... " if path separaters use \ and not /
model = r"C:\Users\PHOENiX\Documents\ESRGAN Models\PSNR_x4_DB.pth"

# 1. Create a VSGAN instance, which creates a PyTorch device instance
vsgan = VSGAN(device)

# 2. Load an ESRGAN model into the VSGAN instance
# tip: You can run load_model() at any point to change the model
vsgan.load_model(model)

# 3. Convert the clip to RGB24 as ESRGAN can only work with linear RGB data
clip = core.resize.Point(clip, format=vs.RGB24)

# 4. Use the VSGAN instance (with its loaded model) on a clip
clip = vsgan.run(clip)

# (optional) Convert back to any other color space if you wish.
# clip = core.resize.Point(clip, format=vs.YUV420P8)

# ...

# Don't forget to set the output clip
clip.set_output()
```
