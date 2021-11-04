import numpy as np
from ImageClasses.UltrasoundScan import UltrasoundScan


def gauss_filter(self, size_x, size_y):
    kernel_x = np.array([1.0, 2.0, 1.0])
    kernel_y = np.array([1.0, 2.0, 1.0])

    filtered_once = np.apply_along_axis(lambda x: np.convolve(x, kernel_x, mode='same'), 0, self.image_3d)
    filtered_twice = np.apply_along_axis(lambda x: np.convolve(x, kernel_y, mode='same'), 1, filtered_once)
    return UltrasoundScan(filtered_twice.astype(dtype=np.uint8))
