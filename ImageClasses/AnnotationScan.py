from NumpyImage import *


class AnnotationLineScan(NumpyImage):
    # Wrapper around each image which is held as a 3d numpy array.

    def __init__(self, rows):
        super().__init__(rows)
        self.data = []

    # has pair
    # convert to points


class AnnotationPointScan(NumpyImage):
    # Wrapper around each image which is held as a 3d numpy array.

    def __init__(self, rows):
        super().__init__(rows)
        self.data = []

    # has pair

