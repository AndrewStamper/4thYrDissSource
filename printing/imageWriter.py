import png
import numpy as np


def write_png(directory, filename, image_3d):
    # Function to read the write numpy array of rows*columns*pixels to a png
    (height, width, colour_depth) = image_3d.shape
    image_2d = np.reshape(image_3d, (-1, width * colour_depth))

    file = open(directory + filename + '.png', 'wb')
    w = png.Writer(width, height, greyscale=False, alpha=True, bitdepth=8)
    w.write(file, image_2d)
    file.close()


def write_jpg():
    print("test")
