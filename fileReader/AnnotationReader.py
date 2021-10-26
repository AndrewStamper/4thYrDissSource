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
