from ImageClasses.TwoDNumpyImage import TwoDNumpyImage
from Constants import *
import numpy as np


class AnnotationMaskScan(TwoDNumpyImage):
    # Class to encapsulate a given mask for illium, femoral head or labrum

    # initialise the structure
    def __init__(self, rows=None, filename=None, scan=None):
        super().__init__(rows, filename)
        self.ultrasound_scan = scan

    def _read_image(self, filename, location=MASK_FILE):
        super()._read_image(filename, location)


# collection of masks one for each structure for a given scan
class MaskCollection:
    def __init__(self, scan_number):
        self.illium = AnnotationMaskScan(filename=(scan_number + "_i.png"))
        self.femoral_head = AnnotationMaskScan(filename=(scan_number + "_f.png"))
        self.labrum = AnnotationMaskScan(filename=(scan_number + "_l.png"))

    def crop(self, shape):
        self.illium.crop(shape)
        self.femoral_head.crop(shape)
        self.labrum.crop(shape)

    def as_RGBA(self):
        i = self.illium.convert_to_rgb(colour=RGBA_RED)
        f_h = self.femoral_head.convert_to_rgb(colour=RGBA_BLUE)
        l = self.labrum.convert_to_rgb(colour=RGBA_GREEN)
        return i + f_h + l

    def as_RGB(self):
        rgba = self.as_RGBA().image_3d
        rgb = rgba[:, :, 0:3]
        return rgb

    def as_segmentation_mask(self):
        rgba = self.as_RGBA().image_3d  # get each mask in separate channel
        rgba[:, :, 3] = 0  # set 4th channel (background) to zeros intensity
        i = [3, 0, 1, 2]  # reorder so if none of three masks are lit it returns background
        argb = rgba[:, :, i]
        pred_mask = np.argmax(argb, axis=-1).reshape((rgba.shape[0], rgba.shape[1], 1))  # brightest of the channels = mask value
        return pred_mask

    def down_sample(self, shape):
        self.illium = self.illium.down_sample(shape)
        self.femoral_head = self.femoral_head.down_sample(shape)
        self.labrum = self.labrum.down_sample(shape)


