from ImageClasses.TwoDNumpyImage import TwoDNumpyImage
from Constants import *
import numpy as np


class AnnotationMaskScan(TwoDNumpyImage):
    # Class to encapsulate a given mask for illium, femoral head or

    # initialise the structure
    def __init__(self, rows=None, filename=None, scan=None):
        super().__init__(rows, filename)
        self.ultrasound_scan = scan

    def _read_image(self, filename, location=MASK_FILE):
        super()._read_image(filename, location)
