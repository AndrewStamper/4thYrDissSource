from PIL import Image
import numpy as np

IMAGE_FILE = "../AlderHayUltrasounds/"


def read_image(filename):
    # Function to read the Annotations stored in the png file and return these as a numpy array of rows*columns*pixels
    img = Image.open(IMAGE_FILE + filename)
    image_3d = np.asarray(img)
    return image_3d


def read_jpg(filename):
    img = Image.open(IMAGE_FILE + filename)
    np_img = np.asarray(img)

    print(np_img.shape)

    pilImage = Image.fromarray(np_img)
    pilImage.save("output/test.png")
    return np_img
