from ImageClasses.TwoDNumpyImage import TwoDNumpyImage
from Constants import *


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
