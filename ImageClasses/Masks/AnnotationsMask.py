from ImageClasses.TwoDNumpyImage import TwoDNumpyImage
from Constants import *
import numpy as np
from PIL import Image


class AnnotationMaskScan(TwoDNumpyImage):
    # Class to encapsulate a given mask for illium, femoral head or labrum

    # initialise the structure
    def __init__(self, rows=None, filename=None, scan=None):
        super().__init__(rows, filename)
        self.ultrasound_scan = scan

    def _read_image(self, filename, location=MASK_FILE):
        super()._read_image(filename, location)

    def write_to_pbm(self, filename, location=OUTPUT_FILE):
        pil_image = Image.fromarray((self.image_3d/255)==0)
        with open(location+filename, 'w') as f:
            f.write(f'P1\n{pil_image.width} {pil_image.height}\n')
            np.savetxt(f, pil_image, fmt='%d')


# collection of masks one for each structure for a given scan
class MaskCollection:
    def __init__(self, scan_number=str(0), generated_mask=None):
        if generated_mask is None:
            self.illium = AnnotationMaskScan(filename=(scan_number + "_i.png"))
            self.femoral_head = AnnotationMaskScan(filename=(scan_number + "_f.png"))
            self.labrum = AnnotationMaskScan(filename=(scan_number + "_l.png"))
        else:
            self.from_segmentation_mask(generated_mask)

    def crop(self, shape):
        self.illium.crop(shape)
        self.femoral_head.crop(shape)
        self.labrum.crop(shape)

    def remove_rightmost(self, x):
        self.illium.remove_rightmost(x)
        self.femoral_head.remove_rightmost(x)
        self.labrum.remove_rightmost(x)

    def as_RGBA(self):
        i = self.illium.convert_to_rgba(colour=RGBA_RED)
        l = self.labrum.convert_to_rgba(colour=RGBA_GREEN)
        f_h = self.femoral_head.convert_to_rgba(colour=RGBA_BLUE)
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

    def from_segmentation_mask(self, generated_mask):
        self.illium = AnnotationMaskScan(rows=(np.array(generated_mask==1).astype(int)*255))
        self.labrum = AnnotationMaskScan(rows=(np.array(generated_mask==2).astype(int)*255))
        self.femoral_head = AnnotationMaskScan(rows=(np.array(generated_mask==3).astype(int)*255))

    def down_sample(self, shape):
        self.illium = self.illium.down_sample(shape)
        self.femoral_head = self.femoral_head.down_sample(shape)
        self.labrum = self.labrum.down_sample(shape)

    def _difference_mask(self, this, other):
        # green is agreed, red is missed blue is miss predict
        agreed = this.convert_to_rgba(colour=RGBA_GREEN).image_3d * np.tile(np.logical_and(this.image_3d > 120, other.image_3d > 120)[:, :, np.newaxis], (1, 1, 4))
        false_neg = this.convert_to_rgba(colour=RGBA_RED).image_3d * np.tile(np.logical_and(this.image_3d > 120, other.image_3d <= 120)[:, :, np.newaxis], (1, 1, 4))
        false_pos = other.convert_to_rgba(colour=RGBA_BLUE).image_3d * np.tile(np.logical_and(this.image_3d <= 120, other.image_3d > 120)[:, :, np.newaxis], (1, 1, 4))
        diff_mask = (agreed + false_neg + false_pos)
        return diff_mask

    def difference_masks(self, other):
        illium = self._difference_mask(self.illium, other.illium)
        femoral_head = self._difference_mask(self.femoral_head, other.femoral_head)
        labrum = self._difference_mask(self.labrum, other.labrum)
        return [illium[:, :, 0:3], femoral_head[:, :, 0:3], labrum[:, :, 0:3]]

    def write_to_pbm(self, filename, location=OUTPUT_FILE):
        self.illium.write_to_pbm("I" + filename, location)
        self.femoral_head.write_to_pbm("F" + filename, location)
        self.labrum.write_to_pbm("L" + filename, location)


