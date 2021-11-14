from ImageClasses.NumpyImage import NumpyImage
from Constants import *
import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go
from functools import partial
import math


class UltrasoundScan(NumpyImage):
    # Wrapper around each scan which is held as a numpy array.

    @staticmethod
    def convert_from_numpy_image(numpy_image):
        # Convert each pixel into a greyscale value
        brightness_matrix = np.repeat([np.repeat([RGB_TO_BRIGHTNESS], numpy_image.get_width(), axis=0)], numpy_image.get_height(), axis=0)
        brightness_image_3d_scan = np.multiply(brightness_matrix, numpy_image.image_3d)
        image_3d = np.sum(brightness_image_3d_scan, axis=2).astype(dtype=np.uint8)
        return UltrasoundScan(image_3d)

    @staticmethod
    # read from a file
    def read_image(filename):
        numpy_image = NumpyImage.read_image(filename)
        return UltrasoundScan.convert_from_numpy_image(numpy_image)

    # initialise the structure
    def __init__(self, rows, annotations_lines=None, annotations_points=None):
        super().__init__(rows)
        self.annotations_lines = annotations_lines
        self.annotations_points = annotations_points

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        numpy_image = super().restrict_to_box(corner_top_left, corner_bottom_right)
        lines = None
        points = None
        if self.annotations_lines is not None:
            lines = self.annotations_lines.restrict_to_box(corner_top_left, corner_bottom_right)
        if self.annotations_points is not None:
            points = self.annotations_points.restrict_to_box(corner_top_left, corner_bottom_right)
        return UltrasoundScan(numpy_image.image_3d, annotations_lines=lines, annotations_points=points)

    # write back to a png
    def write_image(self, filename):
        numpy_image = NumpyImage(self._convert_to_rgb())
        numpy_image.write_image(filename)

    # convert to RGB
    def _convert_to_rgb(self):
        return np.repeat(self.image_3d.reshape((self.get_height(), self.get_width(), 1)), 3, axis=2)

    # plot in 3d
    def plot_as_3d(self, filename):
        x = np.repeat([np.arange(self.get_width())], self.get_height(), axis=0)
        y = np.repeat(np.flip(np.arange(self.get_height())).reshape(-1, 1), self.get_width(), axis=1)

        fig = go.Figure()
        fig.add_trace(go.Surface(z=self.image_3d, x=x, y=y))
        fig.update_layout(
            scene=dict(aspectratio=dict(x=self.get_width() / 100, y=self.get_height() / 100, z=125 / 100), xaxis=dict(range=[0, self.get_width()], ), yaxis=dict(range=[0, self.get_height()])))
        go_offline.plot(fig, filename=OUTPUT_FILE + filename, validate=True, auto_open=False)

    # use a bounding value to segment
    def bound_values(self, brightness):
        bounded_section = np.greater_equal(self.image_3d, np.full(self.get_shape(), brightness)).astype('uint8')
        bounded_image_3d_scan = np.reshape(bounded_section * 255, (bounded_section.shape[0], bounded_section.shape[1], 1))
        return UltrasoundScan(bounded_image_3d_scan)

    # add the points from the annotation to the scan to produce a new image
    def add_annotations(self):
        if self.annotations_points is None:
            raise ValueError('Ultrasound Scan is not Annotated with points')
        scan_image = self._convert_to_rgb()
        mask = np.repeat(np.reshape((self.annotations_points.image_3d[:, :, 3] == 0), (self.get_height(), self.get_width(), 1)), 3, axis=2)
        image_3d_scan_with_points = np.add(np.multiply(scan_image, mask), self.annotations_points.image_3d[:, :, 0:3])
        return NumpyImage(image_3d_scan_with_points)

    def add_progression(self, other):
        if other.get_height() != self.get_height():
            raise ValueError('Cannot add image to progression as it has incompatible height dimension')
        self.image_3d = np.append(self.image_3d, other.image_3d, axis=1)

    def gauss_filter(self, size, standard_deviation):
        if size % 2 == 0:
            raise ValueError("kernel must have odd size")

        kernel = np.zeros(size)
        for i in range(0, size):
            x = i - 1 - size / 2
            kernel[i] = (1 / math.sqrt(2 * math.pi * math.pow(standard_deviation, 2))) * math.exp(-(pow(x, 2) / 2 * pow(standard_deviation, 2)))
        size_1_kernel = kernel / kernel.sum()
        # print(size_1_kernel)

        filtered_once = np.apply_along_axis(lambda x: np.convolve(x, size_1_kernel, mode='same'), 0, self.image_3d)
        filtered_twice = np.apply_along_axis(lambda x: np.convolve(x, size_1_kernel, mode='same'), 1, filtered_once).astype(dtype=np.uint8)
        return UltrasoundScan(filtered_twice)

    def gradient(self, size, high_val='upper_quartile', low_val='lower_quartile'):
        high_func = {
            'max': partial(np.amax, axis=2),
            'mean': partial(np.mean, axis=2),
            'median': partial(np.median, axis=2),
            'upper_quartile': partial(np.quantile, q=0.75, axis=2)
        }[high_val]
        low_func = {
            'min': partial(np.amin, axis=2),
            'mean': partial(np.mean, axis=2),
            'median': partial(np.median, axis=2),
            'lower_quartile': partial(np.quantile, q=0.25, axis=2)
        }[low_val]

        v = self.sliding_window_view((size, size), padding_type='input')
        v = np.reshape(v, (v.shape[0], v.shape[1], -1))

        high = high_func(v)
        low = low_func(v)
        result = high-low
        return UltrasoundScan(result)

    def sliding_window_view(self, shape, step=(1, 1), array=None, padding_type='input', padding=0):
        if array is None:
            input_array = self.image_3d
        else:
            input_array = array

        # TODO fix the padding indexes for even size filters
        # TODO fix padding in output padding mode
        # TODO increment a step feature for stride

        if padding_type == 'input':
            rows_to_add_each_side = math.floor(shape[0] / 2)
            columns_to_add_each_side = math.floor(shape[1] / 2)
            new_shape = (input_array.shape[0] + (rows_to_add_each_side * 2), input_array.shape[1] + (columns_to_add_each_side * 2))
            to_stride = np.full(new_shape, padding)
            to_stride[rows_to_add_each_side:new_shape[0] - rows_to_add_each_side, columns_to_add_each_side:new_shape[1] - columns_to_add_each_side] = input_array

        elif padding_type == 'output' or padding_type == 'none':
            to_stride = input_array
        else:
            raise ValueError('selected padding mode is not supported')

        stride = np.lib.stride_tricks.sliding_window_view(to_stride, shape)

        if padding_type == 'output':
            stride_prime = np.full((*input_array.shape, *shape), padding)
            print(stride_prime.shape)
            print(stride.shape)
            stride_prime[shape[0] - 1:stride_prime.shape[0] - shape[0] + 1, shape[1] - 1:stride_prime.shape[1] - shape[1] + 1] = input_array
            stride = stride_prime

        return stride
