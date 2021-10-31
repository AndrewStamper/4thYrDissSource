from Constants import *
from PIL import Image
import numpy as np

# Base object which is a numpy array, width, height, colours etc
# inherit from that to a scan
# scan has Option[annotations]
# inherit from base to a annotations


class NumpyImage:
    # Wrapper around each image which is held as a 3d numpy array.

    # load it
    @staticmethod
    def read_image(filename):
        # Function to read the Annotations stored in the png file and return these as a numpy array of rows*columns*pixels
        img = Image.open(IMAGE_FILE + filename)
        image_3d = np.asarray(img)
        return NumpyImage(image_3d)

    # initialisation function
    def __init__(self, rows):
        self.image_3d = rows

    # print it
    def write_image(self, filename):
        # Function to read the write numpy array of rows*columns*pixels to a png
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

    # print
    def to_string(self):
        return self.image_3d



