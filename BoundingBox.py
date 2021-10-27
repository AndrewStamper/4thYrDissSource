import numpy as np

RGB_TO_BRIGHTNESS = [0.21, 0.72, 0.07]


def find_box(point1, point2, point3, point4):
    corner_top_left = (min(point1[0], point2[0], point3[0], point4[0]), min(point1[1], point2[1], point3[1], point4[1]))
    corner_bottom_right = (max(point1[0], point2[0], point3[0], point4[0]), max(point1[1], point2[1], point3[1], point4[1]))
    return corner_top_left, corner_bottom_right


def restrict_to_box(image_3d_scan, corner_top_left, corner_bottom_right):
    restricted_image_3d_scan = image_3d_scan[corner_top_left[0]:corner_bottom_right[0], corner_top_left[1]:corner_bottom_right[1]]
    return restricted_image_3d_scan


def brightness_of_pixel(image_3d_scan):
    brightness_matrix = np.repeat([np.repeat([RGB_TO_BRIGHTNESS], image_3d_scan.shape[1], axis=0)], image_3d_scan.shape[0], axis=0)
    brightness_image_3d_scan = np.multiply(brightness_matrix, image_3d_scan)
    return np.sum(brightness_image_3d_scan, axis=2)


def bound_values(image_3d_scan, brightness):
    brightness_image_2d_scan = brightness_of_pixel(image_3d_scan)
    bounded_section = np.greater_equal(brightness_image_2d_scan, np.full(brightness_image_2d_scan.shape, brightness)).astype('uint8')
    bounded_image_3d_scan = np.repeat(np.reshape(bounded_section * 255, (bounded_section.shape[0], bounded_section.shape[1], 1)), 3, axis=2)

    return bounded_image_3d_scan
