---
title: "VSGAN.run(clip: VideoNode, chunk: bool = False)"
permalink: /vsgan-run/
excerpt: "Documentation on the VSGAN.run function."
last_modified_at: 2021-01-16T12:56:00-00:00
toc: false
classes: wide
---

Executes ESRGAN on each frame of the provided clip, returning as a new clip.

- `clip (VideoNode)`: VapourSynth clip (VideoNode) to use.
- `chunk (bool)`: If your system is running out of memory, try enable this as it will split the image into smaller sub-images and render them one by one, then finally merging them back together. Trading memory requirements for speed and accuracy. WARNING: Since the images will be processed separately, the result may have issues on the edges of the chunks, [an example of this issue](https://imgbox.com/g/Hht5NqKB0i).
