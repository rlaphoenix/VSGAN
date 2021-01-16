---
title: "VSGAN.execute(n: int, clip: VideoNode)"
permalink: /vsgan-execute/
excerpt: "Documentation on the VSGAN.execute function."
last_modified_at: 2021-01-16T13:02:00-00:00
toc: false
classes: wide
---

Executes the ESRGAN model on `n`th frame from `clip`. This function is mainly intended to be used internally.
It isn't an error to use this function, but perhaps you're looking for [VSGAN.run]({{ '/vsgan-run/' | absolute_url }}).

- `n (int)`: Frame number.
- `clip (VideoNode)`: Clip to get the frame from.
