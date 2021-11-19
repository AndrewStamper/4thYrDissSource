from ImageClasses.NumpyImage import NumpyImage
from Constants import *
import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go


class UltrasoundScan(NumpyImage):
    # Wrapper around each scan which is held as a numpy array.

    # initialise the structure
    def __init__(self, rows=None, filename=None, annotations_lines=None, annotations_points=None):
        super().__init__(rows, filename)
        if self.image_3d.ndim > 2:
            self.convert_from_numpy_image()
        self.image_3d = self.image_3d - np.amin(self.image_3d)
        self.image_3d = self.image_3d * (255/np.amax(self.image_3d))

        self.annotations_lines = annotations_lines
        self.annotations_points = annotations_points

    # import all the filtering methods
    from ImageClasses.Ultrasound.Filters import gauss_filter, gradient, down_sample, up_sample

    def convert_from_numpy_image(self, numpy_image=None):
        # Convert each pixel into a greyscale value
        if numpy_image is None:
            numpy_image = self

        brightness_matrix = np.repeat([np.repeat([RGB_TO_BRIGHTNESS], numpy_image.get_width(), axis=0)], numpy_image.get_height(), axis=0)
        brightness_image_3d_scan = np.multiply(brightness_matrix, numpy_image.image_3d)
        self.image_3d = np.sum(brightness_image_3d_scan, axis=2).astype(dtype=np.uint8)

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        numpy_image = super().restrict_to_box(corner_top_left, corner_bottom_right)
        lines = None
        points = None
        if self.annotations_lines is not None:
            lines = self.annotations_lines.restrict_to_box(corner_top_left, corner_bottom_right)
        if self.annotations_points is not None:
            points = self.annotations_points.restrict_to_box(corner_top_left, corner_bottom_right)
        return type(self)(numpy_image.image_3d, annotations_lines=lines, annotations_points=points)

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
        return type(self)(bounded_image_3d_scan)

    # add the points from the annotation to the scan to produce a new image
    def add_annotations(self, annotations=None):
        if annotations is None:
            annotations = self.annotations_points
            if annotations is None:
                raise ValueError('Ultrasound Scan is not Annotated with points')
        scan_image = self._convert_to_rgb()
        mask = np.repeat(np.reshape((self.annotations_points.image_3d[:, :, 3] == 0), (self.get_height(), self.get_width(), 1)), 3, axis=2)
        image_3d_scan_with_points = np.add(np.multiply(scan_image, mask), self.annotations_points.image_3d[:, :, 0:3])
        return NumpyImage(image_3d_scan_with_points)

    # add another scan to the side to visualise the progression
    def add_progression(self, other):
        this_image_3d = self.image_3d
        other_image_3d = other.image_3d

        if other.get_height() > self.get_height():
            # must pad this object to add to the progression
            to_add = np.zeros((other.get_height() - self.get_height(), this_image_3d.shape[1]))
            this_image_3d = np.append(this_image_3d, to_add, axis=0)

        if other.get_height() < self.get_height():
            # must pad new object to add to the progression
            to_add = np.zeros((self.get_height() - other.get_height(), other_image_3d.shape[1]))
            other_image_3d = np.append(other_image_3d, to_add, axis=0)

        self.image_3d = np.append(this_image_3d, other_image_3d, axis=1)
