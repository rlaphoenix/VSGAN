# Release History

## 1.4.1

- Reword some error/warning messages, less opinionated, more concise
- Some attributes have been renamed to be more ambiguous in the hopes more Model Architectures get
  supported in the future

**Documentation**

- Create new sphinx documentation. Add much more information, much better structure and readability
- Add HISTORY.md

**Fixes**

- Fix model chaining. It now gets the correct model and model scale values for each FrameEval call
- Fixed the pytorch extra group to correctly be optional and correctly reference a dependency
- Some type-hinting has been corrected

## 1.4.0

- Heavily improve main model execute code
- Add support for all RGB formats including float support to execute
- Replace current chunk system with a seamless chunk system using overlap
- Add self-chaining system, calls can be made directly after another
- Remove .idea folder, add to gitignore

**Fixes**

- Only transpose C for RGB if it's 3-channels

**Documentation**

- Lower-case all references to rlaPHOENiX with rlaphoenix
- Add banner image to README, Right-align colab button, remove unnecessary HRs

**Dependencies**

- Make torch optional, point directly to cuda torch

## 1.3.1

**Fixes**

- Fix type annotations on < 3.9
- Use version 3.9.x for Dist workflow after 3.10 was added to GitHub Actions

## 1.3.0

- Drop support for <= Python 3.6.1, due to bugs discovered in NumPy.
- Rename cv2_imread to frame_to_np, don't reverse to BGR as it's unnecessary
- More efficiently write an array to a vs.VideoFrame for a speed improvement
- Fix bug in relation to frame write array
- Remove the need for plane_count, just get it from the input frame
- Don't define the transposes, it's unnecessary
- Inherit output clip properties from input clip
- Allow specification of the input array dimension order
- Replace setup.py/setuptools with Poetry
- Add Jekyll Documentation in `gh-pages` branch
- Moved README's info, comparison images, and such to the docs
- Create a VSGAN Jupyter Notebook (Colab), with an Open in Colab Badge on the README
- Rework release-packager for auto release+pypi
- Replace deprecated get_read/write_array calls with indexing (Add support for VS API V4 beta)

## 1.2.1

--

## 1.2.0

--

## 1.1.0

**Improvements**

- Replace hardcoded in_nc, out_nc, nf, nb, and scale with values taken directly from the model state.  
  This allows VSGAN to support different kinds of models, like grayscale input, rgb output
  (though untested) as well as smaller models via nf and nb reductions that are faster to train.
  (thanks @JoeyBallentine)
- Very slight QoL improvements, you won't notice these, but hey, cool right?
