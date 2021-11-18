import numpy as np
from functools import partial
import math


# apply a gaussian filter to the scan
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
    return type(self)(filtered_twice)


# do simplistic edge detection using sudden change in brightness
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

    v = _sliding_window_view((size, size), self.image_3d, padding_type='input')
    v = np.reshape(v, (v.shape[0], v.shape[1], -1))

    high = high_func(v)
    low = low_func(v)
    result = high-low
    return type(self)(result)


# down sample a ultrasound image
def down_sample(self, shape):
    return type(self)(_down_sample(shape, self.image_3d, partial(np.mean, axis=2)))


# down sample by a given shape using a given function
def _down_sample(shape, array, function):
    v = _sliding_window_view(shape, array, padding_type='none', step=shape)
    v = np.reshape(v, (v.shape[0], v.shape[1], -1))
    sampled = function(v)
    return sampled


# take a numpy array and produce a sliding window over it
# node that an even size shape will produce an array of different size output to input irrespective of padding
def _sliding_window_view(shape, array, step=(1, 1), padding_type='input', padding=0):

    # TODO implement padding in output padding mode

    if padding_type == 'input':
        rows_to_add_each_side = math.floor(shape[0] / 2)
        columns_to_add_each_side = math.floor(shape[1] / 2)
        new_shape = (array.shape[0] + (rows_to_add_each_side * 2), array.shape[1] + (columns_to_add_each_side * 2))
        to_stride = np.full(new_shape, padding)
        to_stride[rows_to_add_each_side:new_shape[0] - rows_to_add_each_side, columns_to_add_each_side:new_shape[1] - columns_to_add_each_side] = array
    elif padding_type == 'output' or padding_type == 'none':
        to_stride = array
    else:
        raise ValueError('selected padding mode is not supported')

    stride = np.lib.stride_tricks.sliding_window_view(to_stride, shape)

    if padding_type == 'output':
        raise ValueError('output mode padding unsupported')

    if step != (1, 1):
        row_sampled = stride[::shape[0], :, :, :]
        new_stride = row_sampled[:, ::shape[1], :, :]
        stride = new_stride

    return stride

