from PIL import Image
import numpy as np

OUTPUT_FILE = "../output/"


def write_image(filename, image_3d):
    # Function to read the write numpy array of rows*columns*pixels to a png
    file = open(OUTPUT_FILE + filename, 'wb')
    pilImage = Image.fromarray(image_3d)
    pilImage.save(file)
    file.close()


def print_points_png(point1, point2, point3, point4, image_3d, size):
    output_image = np.zeros(image_3d.shape, dtype=np.uint8)
    print_point_for_png(output_image, point1, 0, size)
    print_point_for_png(output_image, point2, 1, size)
    print_point_for_png(output_image, point3, 2, size)
    print_point_for_png(output_image, point4, 0, size)
    print_point_for_png(output_image, point4, 1, size)
    write_image("points.png", output_image)


def print_point_for_png(output_image, point, colour, size):
    for x in range(-size, size):
        for y in range(-size, size):
            if(x * x) + (y * y) <= size*size:
                output_image[point[0] + x, point[1] + y, colour] = 255
                output_image[point[0] + x, point[1] + y, 3] = 255


def write_jpg():
    print("test")
