#####################################################################
#                         Created by PRAGMA                         #
#                 https://github.com/imPRAGMA/VSGAN                 #
#####################################################################
# Dependencies:                                                     #
# - PIP Module: numpy                                               #
# - RRDBNet_arch.py from the xinntao's ESRGAN repo                  #
#   (https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py) #
# - PyTorch: https://pytorch.org/get-started/locally                #
#####################################################################
#              For more details, consult the README.md              #
#####################################################################

from vapoursynth import core
import vapoursynth as vs
import numpy as np
import functools
import torch

# - Torch&Cuda, RRDBNet Arch, and Model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None

# - Start VSGAN operations
def Start(clip, model, scale, old_arch=False):
    global MODEL
    # select the arch to be used based on old_arch parameter
    if old_arch:
        from . import RRDBNet_arch_old as arch
        MODEL = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=scale, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
    else:
        from . import RRDBNet_arch as arch
        MODEL = arch.RRDBNet(3, 3, 64, 23, gc=32)
    # load the model with selected arch
    MODEL.load_state_dict(torch.load(model), strict=True)
    MODEL.eval()
    # tie model to pytorch device
    MODEL = MODEL.to(DEVICE)
    # remember the clip's original format
    orig_format = clip.format
    # convert clip to RGB24 as it cannot read any other color space
    buffer = core.resize.Point(clip, format=vs.RGB24)
    # take a frame when being used by VapourSynth and send it to the execute function
    # returns the edited frame in a 1 frame clip based on the trained model
    buffer = core.std.FrameEval(
        core.std.BlankClip(
            buffer,
            width=clip.width*scale,
            height=clip.height*scale
        ),
        functools.partial(
            Execute,
            clip=buffer
        )
    )
    # Convert back to the original color space and return it to sender
    return core.resize.Point(buffer, format=orig_format, matrix_s="709") #should matrix be gotten from original clip?

# - Deals with the number crunching
def Execute(n, clip):
    # get the frame being used
    frame = clip.get_frame(n)
    # convert it to a numpy readable array
    def frameToNumpyArray(frame, planes):
        return np.dstack([np.array(frame.get_read_array(i), copy=False) for i in reversed(range(planes))])
    numpy_array = frameToNumpyArray(frame, clip.format.num_planes) #num_planes is expected to always be 3
    # use the model's trained data against the images planes
    with torch.no_grad():
      s = MODEL(torch.from_numpy(np.transpose((numpy_array * 1.0 / 255)[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(DEVICE)).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    s = (np.transpose(s[[2, 1, 0], :, :], (1, 2, 0)) * 255.0).round()
    if len(s.shape) <= 3: s = s.reshape([1] + list(s.shape))
    planes = s.shape[-1]
    def conv(n, f):
      fout = f.copy()
      idx = -1
      for p in reversed(range(planes)):
        idx += 1
        d = np.array(fout.get_write_array(p), copy=False)
        np.copyto(d, s[n, :, :, idx], casting="unsafe")
        del d
      return fout
    # create a blank clip with the plane types returned by the model
    clip = core.std.BlankClip(None, s.shape[-2], s.shape[-3], vs.RGB24, s.shape[-4])
    # take the blank clip and insert the new data into the planes (reversed) and return it back to sender
    return core.std.ModifyFrame(clip, clip, conv)