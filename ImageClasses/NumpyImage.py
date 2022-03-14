from Constants import *
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


class NumpyImage:

    """
    Wrapper around each image which is held as a 3d numpy array.
    Base object which is a numpy array, width, height, colours etc
    """

    @staticmethod
    def find_box(points):
        """text
        """
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
        directory_path = os.path.dirname(os.path.abspath(__file__))
        new_path = os.path.join(directory_path, "../" + location + filename)
        img = Image.open(new_path)
        self.image_3d = np.asarray(img, dtype=np.uint8)

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        restricted_image_3d_scan = self.image_3d[corner_top_left[0]:corner_bottom_right[0], corner_top_left[1]:corner_bottom_right[1]]
        return type(self)(restricted_image_3d_scan)

    # crop an image to a selected size
    def crop(self, shape):
        # print("image dimensions: " + str(self.image_3d.shape))
        difference = np.subtract(self.image_3d.shape, shape)
        if not(np.all(difference > 0)):
            print("image dimensions: " + str(self.image_3d.shape))
            print("crop dimensions: " + str(shape))
            raise ValueError("cropping to a image size larger than the original image")

        corner_top_left = np.int64(np.floor(difference/2))
        corner_bottom_right = np.subtract(self.image_3d.shape, difference - corner_top_left)
        self.image_3d = self.restrict_to_box(corner_top_left, corner_bottom_right).image_3d

    # print it
    def write_image(self, filename, image=None):
        if image is None:
            image = self.image_3d

        # Function to read the write numpy array of rows*columns*pixels to a png
        if filename[-4:] != '.png':
            filename = filename+'.png'

        directory_path = os.path.dirname(os.path.abspath(__file__))
        new_path = os.path.join(directory_path, "../" + OUTPUT_FILE + filename)

        file = open(new_path, 'wb')
        pil_image = Image.fromarray(self.image_3d)
        pil_image.save(file)
        file.close()

    def __add__(self, other):
        if self.get_shape() != other.get_shape():
            raise ValueError('invalid size of masks to combine')
        return type(self)(self.image_3d + other.image_3d)

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

    def plot(self, image=None, title="Image_3d"):
        if image is None:
            image = self.image_3d

        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        plt.figure(figsize=(15, 15))
        plt.title('title')
        plt.imshow(tf.keras.utils.array_to_img(image))
        plt.axis('off')
        plt.show()
