#####################################################################
#                         Created by PRAGMA                         #
#                 https://github.com/imPRAGMA/VSGAN                 #
#####################################################################
#              For more details, consult the README.md              #
#####################################################################

import functools

import mvsfunc
import numpy as np
import torch
import vapoursynth as vs
from vapoursynth import core


class VSGAN:

    def __init__(self, device="cuda"):
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Stubs
        self.model_file = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model, scale):
        self.model_file = model
        self.model_scale = scale
        # attempt to use New Arch, and if that fails, attempt to use Old Arch
        # if both fail to be loaded, it will raise it's original exception
        for arch in range(2):
            self.rrdb_net_model = self.get_rrdb_net_arch(arch)
            try:
                self.rrdb_net_model.load_state_dict(torch.load(self.model_file), strict=True)
                break
            except RuntimeError:
                if arch == 1:
                    raise
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)

    def run(self, clip, chunk=False):
        # remember the clip's original format
        original_format = clip.format
        # convert clip to RGB24 as it cannot read any other color space
        buffer = mvsfunc.ToRGB(clip, depth=8)  # expecting RGB24 8bit
        # send the clip array to execute()
        results = []
        for c in self.chunk_clip(buffer) if chunk else [buffer]:
            results.append(core.std.FrameEval(
                core.std.BlankClip(
                    clip=c,
                    width=c.width * self.model_scale,
                    height=c.height * self.model_scale
                ),
                functools.partial(
                    self.execute,
                    clip=c
                )
            ))
        # if chunked, rejoin the chunked clips otherwise return the result
        buffer = core.std.StackHorizontal([
            core.std.StackVertical([results[0], results[1]]),
            core.std.StackVertical([results[2], results[3]])
        ]) if chunk else results[0]
        
        # VSGAN used to convert back to the original color space which resulted
        # in a LOT of guessing, which was in-accurate and may not be efficient
        # depending on what the user is doing after running VSGAN, so in all
        # versions after 1.0.6-post1 we return it in the colorspace the GAN
        # provides which is always RGB24.

        # return the new frame
        return buffer

    def get_rrdb_net_arch(self, arch):
        """
        Import Old or Current Era RRDB Net Architecture
        """
        if arch == 0:
            from . import RRDBNet_arch_old as Arch
            return Arch.RRDB_Net(
                3, 3, 64, 23,
                gc=32,
                upscale=self.model_scale,
                norm_type=None,
                act_type="leakyrelu",
                mode="CNA",
                res_scale=1,
                upsample_mode="upconv"
            )
        from . import RRDBNet_arch as Arch
        return Arch.RRDBNet(3, 3, 64, 23, gc=32)

    @staticmethod
    def chunk_clip(clip):
        # split the clip horizontally into 2 images
        crops = {
            "left": core.std.CropRel(clip, left=0, top=0, right=clip.width / 2, bottom=0),
            "right": core.std.CropRel(clip, left=clip.width / 2, top=0, right=0, bottom=0)
        }
        # split each of the 2 images from above, vertically, into a further 2 images (totalling 4 images per frame)
        # top left, bottom left, top right, bottom right
        return [
            core.std.CropRel(crops["left"], left=0, top=0, right=0, bottom=crops["left"].height / 2),
            core.std.CropRel(crops["left"], left=0, top=crops["left"].height / 2, right=0, bottom=0),
            core.std.CropRel(crops["right"], left=0, top=0, right=0, bottom=crops["right"].height / 2),
            core.std.CropRel(crops["right"], left=0, top=crops["right"].height / 2, right=0, bottom=0)
        ]

    @staticmethod
    def cv2_imread(frame, plane_count):
        """
        Alternative to cv2.imread() that will directly read images to a numpy array
        """
        return np.dstack(
            [np.array(frame.get_read_array(i), copy=False) for i in reversed(range(plane_count))]
        )

    @staticmethod
    def cv2_imwrite(image, out_color_space="RGB24"):
        """
        Alternative to cv2.imwrite() that will convert the data into an image readable by VapourSynth
        """
        if len(image.shape) <= 3:
            image = image.reshape([1] + list(image.shape))
        # Define the shapes items
        plane_count = image.shape[-1]
        image_width = image.shape[-2]
        image_height = image.shape[-3]
        image_length = image.shape[-4]
        # this is a clip (or array buffer for frames) that we will insert the GAN'd frames into
        buffer = core.std.BlankClip(
            clip=None,
            width=image_width,
            height=image_height,
            format=vs.PresetFormat[out_color_space],
            length=image_length
        )

        def replace_planes(n, f):
            frame = f.copy()
            for i, plane_num in enumerate(reversed(range(plane_count))):
                # todo ; any better way to do this without storing the np.array in a variable?
                # todo ; perhaps some way to directly copy it to s?
                d = np.array(frame.get_write_array(plane_num), copy=False)
                # copy the value of d, into s[frame_num, :, :, plane_num]
                np.copyto(d, image[n, :, :, i], casting="unsafe")
                # delete the d variable from memory
                del d
            return frame

        # take the blank clip and insert the new data into the planes and return it back to sender
        return core.std.ModifyFrame(clip=buffer, clips=buffer, selector=replace_planes)

    def execute(self, n, clip):
        """
        Essentially the same as ESRGAN, except it replaces the cv2 functions with ones geared towards VapourSynth
        https://github.com/xinntao/ESRGAN/blob/master/test.py#L26
        """
        # get the frame being used
        frame = clip.get_frame(n)
        img = self.cv2_imread(frame=frame, plane_count=clip.format.num_planes)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lr = img.unsqueeze(0)
        img_lr = img_lr.to(self.torch_device)
        with torch.no_grad():
            output = self.rrdb_net_model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        return self.cv2_imwrite(image=output, out_color_space=clip.format.name)
