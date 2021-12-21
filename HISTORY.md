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

TODO

## 1.2.0

TODO

## 1.1.0

- Add GitHub Actions for build tests and distributions

**Fixes**

- Replace hardcoded `in_nc`, `out_nc`, `nf`, `nb`, and `scale` with values taken directly from the model state (@JoeyBallentine)
- Ensure a model has been loaded before `execute` can be called

**Documentation**

- Replace all references of old username to new username, fixing links and badges
- Update the year in the LICENSE
- Remove pointless `Created by` block of comment at the top of the main python file
- Move majority of documentation and info from the GitHub Wikis system to the README

## 1.0.8

- Change the RGB conversion check's kernel to Spline36

## 1.0.7

**Fixes**

- Get rid of the color space conversion implemented in v1.0.3 as it can be a lossy operation
  The colorspace returned (RGB) may be wanted anyhow, or the user may want to convert a different
  way, if at all (or yet)
- Replace unsafe assert in RRDBNet with if+raise as asserts may be removed when optimised as python byte code files

**Documentation**

- Fix Wiki link in README
- Add some project badges to the README, starting to spice it up now
- Add information on the project, and ESRGAN, with some of ESRGANs comparison images

## 1.0.6

- Detect ESRGAN old/new arch models via archaic trial-and-error

**Mistakes**

- (Kinda) Accidentally added JetBrains PyCharm `.idea` project folder to repo
- A post1 re-version had to be made due to a mistake in the for loop range

## 1.0.5

- Rework code from Functional to Object-oriented Programming
- Improve code readability, project starting to get serious

## 1.0.4

- Add initial method of chunk-mode (does not hide seams)

## 1.0.3

- Convert back to original colorspace after execution

**Dependencies**

- Add VapourSynth to requirements

## 1.0.2

- Add ability to select device via argument

## 1.0.1

- Improve RGB conversion using mvsfunc instead of `core.resize.Point`

**Documentation**

- Add a README file with some basic information

## 1.0.0

Initial Release
