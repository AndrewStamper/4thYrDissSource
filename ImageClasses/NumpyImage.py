from Constants import *
from PIL import Image
import numpy as np


# Wrapper around each image which is held as a 3d numpy array.
# Base object which is a numpy array, width, height, colours etc
class NumpyImage:

    @staticmethod
    def find_box(points):
        corner_top_left = (np.amin(points[:, 0]), np.amin(points[:, 1]))
        corner_bottom_right = (np.amax(points[:, 0]), np.amax(points[:, 1]))
        return corner_top_left, corner_bottom_right

    # initialisation function
    def __init__(self, rows=None, filename=None):
        if rows is not None:
            self.image_3d = rows.astype(dtype=np.uint8)
        elif filename is not None:
            self._read_image(filename)
        else:
            raise ValueError('No initialisation given')

    # Function to read the image stored in the png file and return these as a numpy array of rows*columns*pixels
    def _read_image(self, filename, location=IMAGE_FILE):
        img = Image.open(location + filename)
        self.image_3d = np.asarray(img, dtype=np.uint8)

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        restricted_image_3d_scan = self.image_3d[corner_top_left[0]:corner_bottom_right[0], corner_top_left[1]:corner_bottom_right[1]]
        return type(self)(restricted_image_3d_scan)

    # print it
    def write_image(self, filename):
        # Function to read the write numpy array of rows*columns*pixels to a png
        if filename[-4:] != '.png':
            filename = filename+'.png'

        file = open(OUTPUT_FILE + filename, 'wb')
        pil_image = Image.fromarray(self.image_3d)
        pil_image.save(file)
        file.close()

    # width
    def get_width(self):
        return self.image_3d.shape[1]

    # height
    def get_height(self):
        return self.image_3d.shape[0]

    # colour dimensionality
    def get_colour_depth(self):
        return self.image_3d.shape[2]

    # shape
    def get_shape(self):
        return self.image_3d.shape

    # print
    def to_string(self):
        return self.image_3d
