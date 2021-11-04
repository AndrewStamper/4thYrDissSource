from ImageClasses.NumpyImage import *


class AnnotationPointScan(NumpyImage):
    # Wrapper around point annotation of the scan which is held as a 3d numpy array.

    # initialise the structure
    def __init__(self, rows, points, scan=None, lines=None):
        super().__init__(rows)
        self.points = points
        self.ultrasound_scan = scan
        self.annotations_lines = lines
        if self.ultrasound_scan is not None:
            self.ultrasound_scan.annotations_points = self

    # take a sub-image from the whole image
    def restrict_to_box(self, corner_top_left, corner_bottom_right):
        numpy_image = super().restrict_to_box(corner_top_left, corner_bottom_right)
        new_points = self.points  # TODO move these points
        return AnnotationPointScan(numpy_image.image_3d, new_points)

    def illium_l_point(self):
        return self.points[0]  # TODO setup as right point

    def illium_r_point(self):
        return self.points[0]  # TODO setup as right point

    def bony_roof_b_point(self):
        return self.points[0]  # TODO setup as right point

    def bony_roof_t_point(self):
        return self.points[0]  # TODO setup as right point
