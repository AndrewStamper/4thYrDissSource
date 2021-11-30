from ImageClasses.TwoDNumpyImage import TwoDNumpyImage, NumpyImage
import numpy as np


class UltrasoundScan(TwoDNumpyImage):
    # Wrapper around each scan which is held as a numpy array.

    # initialise the structure
    def __init__(self, rows=None, filename=None, annotations_lines=None, annotations_points=None):
        super().__init__(rows, filename)

        self.annotations_lines = annotations_lines
        self.annotations_points = annotations_points

    # import all the filtering methods
    from ImageClasses.Ultrasound.Filters import gauss_filter, gradient, down_sample, up_sample

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

    # add the points from the annotation to the scan to produce a new image
    def add_annotations(self, annotations=None):
        if annotations is None:
            annotations = self.annotations_points
            if annotations is None:
                raise ValueError('Ultrasound Scan is not Annotated with points')
        scan_image = self._convert_to_rgb()
        mask = np.repeat(np.reshape((annotations.image_3d[:, :, 3] == 0), (self.get_height(), self.get_width(), 1)), 3, axis=2)
        image_3d_scan_with_points = np.add(np.multiply(scan_image, mask), annotations.image_3d[:, :, 0:3])
        return NumpyImage(image_3d_scan_with_points)
