from NumpyImage import *
from Constants import *
import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go


class UltrasoundScan(NumpyImage):
    # Wrapper around each scan which is held as a numpy array.

    @staticmethod
    def convert_from_numpy_image(numpy_image):
        # Convert each pixel into a greyscale value
        brightness_matrix = np.repeat([np.repeat([RGB_TO_BRIGHTNESS], numpy_image.get_width(), axis=0)], numpy_image.get_height(), axis=0)
        brightness_image_3d_scan = np.multiply(brightness_matrix, numpy_image.image_3d)
        image_3d = np.sum(brightness_image_3d_scan, axis=2)
        return UltrasoundScan(image_3d)

    @staticmethod
    # read from a file
    def read_image(filename):
        numpy_image = super().read_image(filename)
        return UltrasoundScan.convert_from_numpy_image(numpy_image)

    # initialise the structure
    def __init__(self, rows):
        super().__init__(rows)
        self.annotations = None

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        numpy_image = super().restrict_to_box(corner_top_left, corner_bottom_right)
        return UltrasoundScan(numpy_image.image_3d)

    # set the annotations
    def set_annotations(self, pair):
        self.annotations = pair

    # retrieve the annotations
    def get_annotations(self):
        return self.annotations

    # plot in 3d
    def plot_as_3d(self, filename):
        x = np.repeat([np.arange(self.get_width())], self.get_height(), axis=0)
        y = np.repeat(np.flip(np.arange(self.get_height())).reshape(-1, 1), self.get_width(), axis=1)

        fig = go.Figure()
        fig.add_trace(go.Surface(z=self.image_3d, x=x, y=y))
        fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5), xaxis=dict(range=[0, self.get_width()],), yaxis=dict(range=[0, self.get_height()])))
        go_offline.plot(fig, filename=OUTPUT_FILE + filename, validate=True, auto_open=False)

    # use a bounding value to segment
    def bound_values(self, brightness):
        bounded_section = np.greater_equal(self.image_3d, np.full(self.get_shape(), brightness)).astype('uint8')
        bounded_image_3d_scan = np.repeat(np.reshape(bounded_section * 255, (bounded_section.shape[0], bounded_section.shape[1], 1)), 3, axis=2)
        return UltrasoundScan(bounded_image_3d_scan)
