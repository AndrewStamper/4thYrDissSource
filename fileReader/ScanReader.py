# LEGACY image loader


from PIL import Image
import numpy as np

IMAGE_FILE = "../AlderHayUltrasounds/"


def read_image(filename):
    # Function to read the Annotations stored in the png file and return these as a numpy array of rows*columns*pixels
    img = Image.open(IMAGE_FILE + filename)
    image_3d = np.asarray(img)
    return image_3d
