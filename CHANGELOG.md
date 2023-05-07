# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added support for Python 3.11.
- Added pre-commit hooks and config, as well as flake8 and isort for use during development.
- Wrote a Contributing bible to CONTRIBUTING.md.
- Added support for two more model architectures:
  [HAT](https://arxiv.org/abs/2205.04437) and [SwinIR](https://arxiv.org/abs/2108.10257).

### Changed

- The structural definition of model, arch, networks, and blocks have been refreshed in both the files and docs.
  Effectively the code between the `/archs` and `/networks` folders have been flipped to be more correct.
  If you imported via the `networks` sub-package, e.g., `from vsgan.networks import ESRGAN`, then please update the
  import path to `from vsgan.archs import ESRGAN` or just `from vsgan import ESRGAN`.
- All previous networks (previously under the `/archs` folder) have been renamed from the name of the Architecture
  they were used for, to their actual network name. E.g., `ESRGAN.py` with the class `ESRGAN(nn.Module)` is now more
  appropriately named `rrdb.py` with the class `RRDBNet(nn.Module)`.
- The minimum supported PyTorch version is now v1.12.0. This is still a fairly old version so compatibility with other
  tools and environments should still be the same.
- The utility `recursive_tile_tensor` has been renamed to `tile_tensor_r` to reduce line length.
- The `load` method of Network's have has the `model` parameter renamed to `state`. The doc-strings have also been
  reflected. This is to reflect more closely to what the "model" parameter actually is, a state file (.pth), not a
  "model" per-say.
- The `device` and `model` class instance variables have been privatized to `_device` and `_model`.
  This is to discourage access to them externally, especially manually altering their value.
- Reversed the following change in v1.6.4: [`Now directly loads a tensor from the VideoFrame data directly, without
  numpy as a middleman.`](https://github.com/rlaphoenix/VSGAN/commit/a49e844d6a6a2edaa588f6e012a24fdaf265c4d6).
  This is because the method used in that change doesn't seem to have any performance benefits, but causes torch to
  complain about the tensor not being writable. This doesn't seem to be problematic for me on Windows, but it might
  be on others (perhaps linux? mac?).

### Removed

- Dropped support for Python 3.7 (therefore, also dropping support for VapourSynth R48).
- Removed unused block `ConcatBlock` which could be used to concat the input with the output of a submodule.
- Removed unused activation initialization when constructing SRVGGNetCompact (Real-ESRGAN v2) Models.
  The same activation was initialized per conv layer instead.
- Removed unused function `float32_to_uint8` from EGVSR which could be used to cast numpy arrays of precision
  float32 to uint8.

### Fixed

- Fixed loading error of some model files if the `.pth` file had the params data moved from the `params`/`params_ema`
  key to the root. E.g., chaiNNer's interpolate models feature.
- Clamped VapourSynth frames after being converted to a tensor as they would be slightly out of the 0,1 range bounds.
  This fixes an issue with EGVSR where some tensors would be -inf instead of 0.0, as well as small weird issues in the
  resulting frame.
- Fixed an edge-case when chaining models by using a UUID4 as a run identifier instead of the depth cache length.
- Fixed VapourSynth dependency markers version matrix to correctly use the latest versions for each Python version
  instead of asking for R57 on all versions, including those with only support for older versions. For example, Python
  3.7 only supports VapourSynth R48 and nothing newer, so asking for R57 on 3.7 wouldn't be possible.
- Added missing future annotations import on architectures, fixing support for Python 3.7 (which support for has since
  been dropped).
- Various issues during Docs building procedure has been squashed. ReadTheDocs now correctly builds the entirety of
  the docs without needing VapourSynth installed (which isn't currently possible to do on their build runners). This
  includes fixing the Interface (api) page which was previously blank due to it failing to build that page.

## [1.6.4] - 2022-01-25

### Changed

- Renamed GitHub Workflows from Distribution to CD and Version test to CI.
- Now caching the VapourSynth installation in GitHub CI workflow.
- Now directly loads a tensor from the VideoFrame data directly, without numpy as a middleman.
- Reduced overall numpy use for Tensor<->VideoFrame operations.
- The half parameter/options have been removed entirely and replaced with automatic infer based on
  input bit-depth. You must explicitly use RGBS if you want FullTensor (float32). Integer and RGBH
  inputs will be converted (if needed) to float16 (HalfTensor) automatically.

### Removed

- Removed EOL Python 3.6 from CI Workflow.
- Removed unused infer_sequence method from EGVSR arch.
- Removed unused options and code from frame_to_tensor.
- All manual tensor deletion statements have been removed, they do not seem to help with VRAM.
- The overlap reduction code per-recursion has been removed. The overlap will now always stay
  at the value first provided.

### Fixed

- Fixed a big Memory leak, that I still don't know exactly why it happened.
- Fixed minimum Python version listed under Installation docs.
- Improved the accuracy of clamping max size value to an equation on the exact bit depth.
  This fixes the accuracy of RGB 27, 30, 36, and 42.

## [1.6.3] - 2022-01-24

### Added

- Recursive tiling depth is now cached per-clip, rather than per-frame.

### Changed

- Updated numpy to version 1.21.1.

### Removed

- Dropped support for Python versions older than 3.7.

### Fixed

- Fix another regression with rejoined tensors defaulting creation on the default device.

## [1.6.2] - 2022-01-24

### Fixed

- Fix another regression due to incorrect overlap scaling calculation from within `join_tiles()`.

## [1.6.1] - 2022-01-24

### Fixed

- Fix regression due to missing overlap specification to `join_tiles()` from within `recursive_tile_tensor()`.

## [1.6.0] - 2022-01-24

### Added

- Add support for [EGVSR](https://arxiv.org/abs/2107.05307), Arch and Network.
- Add support for [Real-ESRGAN-v2](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)
  aka Anime Video Models (comp. vgg-style arch).
- Ability to use half-precision (fp16, HalfTensor) via `half` parameter. This can help reduce VRAM.
- Created tiling utilities to tile a tensor, merge tiled tensors, and automatically tile and execute
  recursively.

### Changed

- Moved the frame/numpy/tensor utility functions out of the VSGAN class and into `utilities.py`.
- Renamed HISTORY to CHANGELOG, and updated changelog to be in Keep a Changelog standard.
- Moved VSGAN class from `__init__.py` to `vsgan.py`.
- Tiling mode is now always enabled, but will only tile if you wouldn't have otherwise had enough VRAM.
- Overlap now defaults to 16.
- Separated VSGAN class into two separate Network classes, ESRGAN, and EGVSR. VSGAN is no longer used
  and ESRGAN/EGVSR Network classes should now be imported and used instead.
- The functions `load_model` and `run` have been renamed to `load` and `apply`.

### Fixed

- Don't require batch in tensor_to_clip.
- Make change_order False by default in frame_to_tensor, improve rest of the param defaults.
- Don't change order to (2,0,1) for ESRGAN models, was unnecessary and caused issues with Real-ESRGANv2.
- Fixed support for Python versions older than 3.8.
- Fixed example VapourSynth import paths casing.
- Restore support for VapourSynth API 3.
- Now detaches tiles from the GPU after super-resolution, to keep space for the next tile's super-resolution.

## [1.5.0] - 2021-12-24

### Added

- Add support for ESRGAN+ models, Real-ESRGAN models (including 2x and 1x if pixel-shuffle was used),
  and A-ESRGAN models.
- Add support for Newer-New-arch in ESRGAN new-to-old state dict conversion.

### Changed

- Rework model/arch file system structure to /models, /models/blocks and /models/ESRGAN.
- Rework ESRGAN architecture as a singular class, with all ESRGAN-specific operation done within it.
- Move ESRGAN-specific blocks within ESRGAN.py.

### Removed

- Removed some unused blocks from RRDBNet.

### Fixed

- Ensure `clip` parameter of VSGAN is a VapourSynth VideoNode object (a clip).
- Move RGB clip check to the constructor of VSGAN rather than `run()`.

## [1.4.1] - 2021-12-21

### Added

- Created new sphinx documentation, replacing the old Jekyll documentation.
- Added HISTORY.md file for recording history (now CHANGELOG.md).

### Changed

- Reword some error/warning messages, now less opinionated and more concise.
- Some attributes have been renamed to be more ambiguous in the hopes more Model Architectures get
  supported in the future.

### Fixed

- Fix model chaining. It now gets the correct model and model scale values for each FrameEval call.
- Fixed the pytorch extra group to correctly be optional and correctly reference a dependency.
- Some type-hinting has been corrected.

## [1.4.0] - 2021-12-13

### Added

- Added support for all RGB formats including float.

### Changed

- Heavily improved main model execution code.
- Replace current chunk system with a seamless chunk system using overlap.
- Add self-chaining system, calls can be made directly after another.
- Made torch dependency optional and pointed directly to torch+cuda.
  This is due to conflicting kinds of torch installation methods.

### Removed

- Remove JetBrains `.idea` folder, added to gitignore.

### Fixed

- Only transpose C for RGB if it's 3-channels.

## [1.3.1] - 2021-10-25

### Fixed

- Fix type annotations on Python versions older than 3.9.
- Use Python version 3.9.x for Dist workflow as 3.10 is not yet supported.

## [1.3.0] - 2021-10-07

### Added

- Allow specification of the input array dimension order.
- Add Jekyll Documentation in `gh-pages` branch.
- Added a VSGAN Jupyter Notebook (Colab), with an Open in Colab Badge on the README.

### Changed

- Drop support for Python versions older than 3.6.2, due to bugs discovered in NumPy.
- Replace setup.py/setuptools with Poetry.
- Rename `cv2_imread` to `frame_to_np`, don't reverse to BGR as it's unnecessary.
- More efficiently write an array to a VapourSynth VideoFrame.
- Inherit output clip properties from input clip.
- Moved README's information to the docs.
- Reworked the CD GitHub Workflow to auto-create a GitHub Release and push to PyPI.

### Removed

- Remove the need for plane_count, now gets it from the input frame.
- Don't define the transposes, it's unnecessary.

### Fixed

- Fixed a bug with frame plane access on VapourSynth API 4.

## [1.2.1] - 2020-12-27

### Added

- Add ability to check what the last loaded model is via `VSGAN.model` attribute.

## [1.2.0] - 2020-12-27

### Added

- Added type-hinting across the code base as well as some doc-strings.

### Changed

- A heavy warning discouraging the use of your CPU as a PyTorch device was added. Ability to use
  your CPU was hidden but reading the warning explains how to do so.
- Reduced required VapourSynth version to 48 or newer.

### Removed

- Remove the conversion to RGB prior to model execution. RGB is required for the Model, but let
  the user decide how to convert to format, what algorithm, how to deal with matrix, and so on.
- Removed setuptools from dependencies.

### Fixed

- Add a check to ensure input clip is RGB, since auto conversion was removed.
- Add missing documentation on [1.1.0]'s changes to scale and such.

## [1.1.0] - 2020-10-20

### Added

- Added two GitHub Action workflows for CI/CD.

### Changed

- Moved the majority of documentation and info from the GitHub Wikis system to the README.

### Fixed

- Replace hardcoded `in_nc`, `out_nc`, `nf`, `nb`, and `scale` with values taken directly from the model state.
- Check that a model has been loaded before `execute` can be called.

## [1.0.8] - 2019-12-19

### Changed

- Change the RGB conversion check's kernel to `Spline36`.

## [1.0.7] - 2019-11-29

### Removed

- Removed the color-space conversion implemented in [1.0.3] as it can be a lossy operation.
  Let the user decide how/if to convert back to the original format. E.g., what algorithm,
  what matrix, and so on.

### Fixed

- Replaced unsafe assert in `RRDBNet` with an if and raise, as asserts may be removed when
  optimised as python byte code files.

## [1.0.6] - 2019-10-20

### Added

- Detect ESRGAN old/new arch models via archaic trial-and-error.

## [1.0.5] - 2019-10-20

### Changed

- Reworked code from Functional to Object-oriented Programming.
- Improve code readability, project starting to get serious.

## [1.0.4] - 2019-10-20

### Added

- Add ability to tile the input to reduce VRAM (does not hide seams).

## [1.0.3] - 2019-10-20

### Added

- VapourSynth to requirements.

### Changed

- Convert back to original color-space after applying the model.

## [1.0.2] - 2019-10-16

### Added

- Ability to select device via argument.

## [1.0.1] - 2019-10-15

### Added

- README file with some basic information.

### Changed

- Improved RGB conversion by using `mvsfunc` instead of `core.resize.Point`.

## [1.0.0] - 2019-09-21

Initial Release.

[Unreleased]: https://github.com/rlaphoenix/VSGAN/compare/v1.6.4...HEAD
[1.6.4]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.6.4
[1.6.3]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.6.3
[1.6.2]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.6.2
[1.6.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.6.1
[1.6.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.6.0
[1.5.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.5.0
[1.4.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.4.1
[1.4.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.4.0
[1.3.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.3.1
[1.3.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.3.0
[1.2.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.2.1
[1.2.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.2.0
[1.1.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.1.0
[1.0.8]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.8
[1.0.7]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.7
[1.0.6]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.6
[1.0.5]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.5
[1.0.4]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.4
[1.0.3]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.3
[1.0.2]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.2
[1.0.1]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.1
[1.0.0]: https://github.com/rlaphoenix/VSGAN/releases/tag/v1.0.0
