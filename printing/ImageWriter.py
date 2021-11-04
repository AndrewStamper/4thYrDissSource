# LEGACY image Loader incorporated into ImageClass:NumpyImage


from PIL import Image

OUTPUT_FILE = "../output/"


def write_image(filename, image_3d):
    # Function to read the write numpy array of rows*columns*pixels to a png
    file = open(OUTPUT_FILE + filename, 'wb')
    pil_image = Image.fromarray(image_3d)
    pil_image.save(file)
    file.close()
