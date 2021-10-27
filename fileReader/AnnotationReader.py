import numpy as np


def get_green(image_3d):
    green_layer = image_3d[:, :, 1]
    width = green_layer.shape[1]
    height = green_layer.shape[0]

    left_to_right = np.repeat([np.arange(width)], height, axis=0)
    top_to_bottom = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1)
    diagonal_tlbr_increasing = np.add(left_to_right, top_to_bottom)  # diagonal tlbr is the diagonal from top left to bottom right
    diagonal_tlbr_decreasing = np.flip(diagonal_tlbr_increasing)

    green_upper_left_most = np.multiply(diagonal_tlbr_decreasing, green_layer)
    point1 = np.unravel_index(np.argmax(green_upper_left_most, axis=None), green_layer.shape)

    green_lower_right_most = np.multiply(diagonal_tlbr_increasing, green_layer)
    point2 = np.unravel_index(np.argmax(green_lower_right_most, axis=None), green_layer.shape)

    return point1, point2


def get_blue(image_3d):
    blue_layer = image_3d[:, :, 2]
    width = blue_layer.shape[1]
    height = blue_layer.shape[0]

    right_to_left = np.repeat([np.flip(np.arange(width))], height, axis=0)
    top_to_bottom = np.repeat(np.arange(height).reshape(-1, 1), width, axis=1)
    diagonal_trbl_increasing = np.add(right_to_left, top_to_bottom)  # diagonal trbl is the diagonal from top right to bottom left
    diagonal_trbl_decreasing = np.flip(diagonal_trbl_increasing)

    blue_upper_right_most = np.multiply(diagonal_trbl_decreasing, blue_layer)
    point3 = np.unravel_index(np.argmax(blue_upper_right_most, axis=None), blue_layer.shape)

    blue_lower_left_most = np.multiply(diagonal_trbl_increasing, blue_layer)
    point4 = np.unravel_index(np.argmax(blue_lower_left_most, axis=None), blue_layer.shape)

    return point3, point4


def print_points_png(point1, point2, point3, point4, image_3d, size):
    output_image = np.zeros(image_3d.shape, dtype=np.uint8)
    print_point_for_png(output_image, point1, 0, size)
    print_point_for_png(output_image, point2, 1, size)
    print_point_for_png(output_image, point3, 2, size)
    print_point_for_png(output_image, point4, 0, size)
    print_point_for_png(output_image, point4, 1, size)
    return output_image


def print_point_for_png(output_image, point, colour, size):
    for x in range(-size, size):
        for y in range(-size, size):
            if(x * x) + (y * y) <= size*size:
                output_image[point[0] + x, point[1] + y, colour] = 255
                output_image[point[0] + x, point[1] + y, 3] = 255


def add_points(image_3d_scan, image_3d_points):
    width = image_3d_points.shape[1]
    height = image_3d_points.shape[0]

    # mask = np.repeat(np.reshape((1 - (image_3d_points[:, :, 3]) / 255), (height, width, 1)), 3, axis=2)

    mask = np.repeat(np.reshape((image_3d_points[:, :, 3] == 0), (height, width, 1)), 3, axis=2)

    image_3d_scan_with_points = np.add(np.multiply(image_3d_scan, mask), image_3d_points[:, :, 0:3])
    return image_3d_scan_with_points
