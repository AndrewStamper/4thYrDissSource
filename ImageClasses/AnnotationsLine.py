from ImageClasses.AnnotationsPoints import AnnotationPointScan
from ImageClasses.NumpyImage import NumpyImage
from Constants import *
import numpy as np


class AnnotationLineScan(NumpyImage):
    # Wrapper around line annotation of the scan which is held as a 3d numpy array.

    @staticmethod
    # read from a file
    def read_image(filename, scan=None):
        numpy_image = NumpyImage.read_image(filename)
        return AnnotationLineScan(numpy_image.image_3d, scan=scan)

    # initialise the structure
    def __init__(self, rows, scan=None):
        super().__init__(rows)
        self.ultrasound_scan = scan
        self.annotations_points = None
        if self.ultrasound_scan is not None:
            self.ultrasound_scan.annotations_lines = self

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        numpy_image = super().restrict_to_box(corner_top_left, corner_bottom_right)
        return AnnotationLineScan(numpy_image.image_3d)

    # convert to points
    def find_points(self):
        points = self._find_points()
        image_3d = self._produce_point_image(points)
        self.annotations_points = AnnotationPointScan(image_3d, points, self.ultrasound_scan, self)
        return self.annotations_points

    def _find_points(self):
        points = np.repeat([[0, 0]], 4, axis=0)
        green_layer = self.image_3d[:, :, 1]
        blue_layer = self.image_3d[:, :, 2]

        # build matrices increasing in each direction required
        left_to_right = np.repeat([np.arange(self.get_width())], self.get_height(), axis=0)
        right_to_left = np.repeat([np.flip(np.arange(self.get_width()))], self.get_height(), axis=0)
        top_to_bottom = np.repeat(np.arange(self.get_height()).reshape(-1, 1), self.get_width(), axis=1)
        diagonal_top_left_to_bottom_right_increasing = np.add(left_to_right, top_to_bottom)
        diagonal_top_left_to_bottom_right_decreasing = np.flip(diagonal_top_left_to_bottom_right_increasing)
        diagonal_top_right_to_bottom_left_increasing = np.add(right_to_left, top_to_bottom)
        diagonal_top_right_to_bottom_left_decreasing = np.flip(diagonal_top_right_to_bottom_left_increasing)

        # use argmax to extremity points on green line
        green_upper_left_most = np.multiply(diagonal_top_left_to_bottom_right_decreasing, green_layer)
        points[0] = np.unravel_index(np.argmax(green_upper_left_most, axis=None), green_layer.shape)
        green_lower_right_most = np.multiply(diagonal_top_left_to_bottom_right_increasing, green_layer)
        points[1] = np.unravel_index(np.argmax(green_lower_right_most, axis=None), green_layer.shape)
        # use argmax to extremity points on blue line
        blue_upper_right_most = np.multiply(diagonal_top_right_to_bottom_left_decreasing, blue_layer)
        points[2] = np.unravel_index(np.argmax(blue_upper_right_most, axis=None), blue_layer.shape)
        blue_lower_left_most = np.multiply(diagonal_top_right_to_bottom_left_increasing, blue_layer)
        points[3] = np.unravel_index(np.argmax(blue_lower_left_most, axis=None), blue_layer.shape)
        return points

    def _produce_point_image(self, points):
        output_image = np.zeros(self.get_shape(), dtype=np.uint8)
        colours = [RGBA_RED, RGBA_BLUE, RGBA_GREEN, RGBA_YELLOW]

        for index in range(0, points.shape[0]):
            for x in range(-POINT_SIZE, POINT_SIZE):
                for y in range(-POINT_SIZE, POINT_SIZE):
                    if(x * x) + (y * y) <= POINT_SIZE*POINT_SIZE:
                        output_image[points[index, 0] + x, points[index, 1] + y] = colours[index]

        return output_image
