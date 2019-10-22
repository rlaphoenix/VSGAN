#####################################################################
#                         Created by PRAGMA                         #
#                 https://github.com/imPRAGMA/VSGAN                 #
#####################################################################
#              For more details, consult the README.md              #
#####################################################################

from vapoursynth import core
import vapoursynth as vs
import numpy as np
import functools
import mvsfunc
import torch

# - Torch & CUDA, RRDBNet Arch, and Model
DEVICE = None
MODEL = None


# - Start VSGAN operations
def start(clip, model, scale, device="cuda", old_arch=False):
    global DEVICE
    global MODEL
    # Setup a device, use CPU instead if cuda isn't available
    if not torch.cuda.is_available():
        device = 'cpu'
    DEVICE = torch.device(device)
    # select the arch to be used based on old_arch parameter
    if old_arch:
        from . import RRDBNet_arch_old as Arch
        MODEL = Arch.RRDB_Net(
            3, 3, 64, 23,
            gc=32,
            upscale=scale,
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            res_scale=1,
            upsample_mode="upconv"
        )
    else:
        from . import RRDBNet_arch as Arch
        MODEL = Arch.RRDBNet(3, 3, 64, 23, gc=32)
    # load the model with selected arch
    MODEL.load_state_dict(torch.load(model), strict=True)
    MODEL.eval()
    # tie model to PyTorch device
    MODEL = MODEL.to(DEVICE)
    # remember the clip's original format
    original_format = clip.format
    # convert clip to RGB24 as it cannot read any other color space
    buffer = mvsfunc.ToRGB(clip, depth=8)  # expecting RGB24 8bit
    # take a frame when being used by VapourSynth and send it to the execute function
    # returns the edited frame in a 1 frame clip based on the trained model
    buffer = core.std.FrameEval(
        core.std.BlankClip(
            buffer,
            width=clip.width * scale,
            height=clip.height * scale
        ),
        functools.partial(
            execute,
            clip=buffer
        )
    )
    # Convert back to the original color space
    if original_format.color_family != buffer.format.color_family:
        if original_format.color_family == vs.ColorFamily.RGB:
            buffer = mvsfunc.ToRGB(buffer)
        if original_format.color_family == vs.ColorFamily.YUV:
            buffer = mvsfunc.ToYUV(buffer, css=original_format.name[3:6])
    # return the new frame/(s)
    return buffer


# - Deals with the number crunching
def execute(n, clip):
    # get the frame being used
    frame = clip.get_frame(n)
    # convert it to a numpy readable array for PyTorch
    numpy_array = np.dstack(
        [np.array(frame.get_read_array(i), copy=False) for i in reversed(range(clip.format.num_planes))]
    )
    # use the model's trained data against the images planes
    with torch.no_grad():
        s = MODEL(
            torch.from_numpy(
                np.transpose((numpy_array * 1.0 / 255)[:, :, [2, 1, 0]], (2, 0, 1))
            ).float().unsqueeze(0).to(DEVICE)
        ).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    s = (np.transpose(s[[2, 1, 0], :, :], (1, 2, 0)) * 255.0).round()
    if len(s.shape) <= 3:
        s = s.reshape([1] + list(s.shape))
    plane_count = s.shape[-1]  # expecting 3 for RGB24 input

    def replace_planes(n, f):
        frame = f.copy()
        for plane_num, p in enumerate(reversed(range(plane_count))):
            # todo ; any better way to do this without storing the np.array in a variable?
            # todo ; perhaps some way to directly copy it to s?
            d = np.array(frame.get_write_array(p), copy=False)
            # copy the value of d, into s[frame_num, :, :, plane_num]
            np.copyto(d, s[n, :, :, plane_num], casting="unsafe")
            # delete the d variable from memory
            del d
        return frame
    # this is a clip (or array buffer for frames) that we will insert the GAN'd frames into
    clip = core.std.BlankClip(
        clip=None,
        width=s.shape[-2],
        height=s.shape[-3],
        format=vs.PresetFormat[clip.format.name],
        length=s.shape[-4]
    )
    # take the blank clip and insert the new data into the planes and return it back to sender
    return core.std.ModifyFrame(clip=clip, clips=clip, selector=replace_planes)
