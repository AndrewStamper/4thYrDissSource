from ImageClasses.Points.AnnotationsPoints import AnnotationPointScan
from ImageClasses.NumpyImage import NumpyImage
from Constants import *
import numpy as np


class AnnotationMaskScan(NumpyImage):
    # Wrapper around line annotation of the scan which is held as a 3d numpy array.

    # initialise the structure
    def __init__(self, rows=None, filename=None, scan=None, colour=None):
        super().__init__(rows, filename)
        self.ultrasound_scan = scan

        if self.image_3d.ndim < 3:
            new_image_3d = np.zeros((*self.image_3d.shape, 4))
            new_image_3d[self.image_3d > 0] = colour
            self.image_3d = new_image_3d

    def _read_image(self, filename, location=MASK_FILE):
        super()._read_image(filename, location)

    def combine_masks(self, other):
        if self.get_shape() != other.get_shape():
            raise ValueError('invalid size of masks to combine')
        return type(self)(self.image_3d + other.image_3d)
