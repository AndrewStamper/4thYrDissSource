from ImageClasses.NumpyImage import NumpyImage
from Constants import *
import numpy as np
import math

illium_t_l = 0
illium_t_r = 1
labrum = 2
illium_b_t = 3
illium_b_b = 4


class AbhiAnnotationPointScan(NumpyImage):
    # Wrapper around Point annotation of the scan which is held as a 3d numpy array.
    # POINTS are held as indecies to the array. i.e. ORIGIN is top left; (130,56) is 130 rows down, 56 columns across; (100,40) is above (150,40); (100,40) is left of (100,50)

    # initialise the structure
    def __init__(self, rows=None, filename=None):
        super().__init__(rows, filename)
        self.points = self._find_points()

    def _read_image(self, filename, location=MASK_FILE):
        super()._read_image(filename, location)

    def illium_t_l_point(self):
        return self.points[illium_t_l]

    def illium_t_r_point(self):
        return self.points[illium_t_r]

    def labrum_point(self):
        return self.points[labrum]

    def illium_b_t_point(self):
        return self.points[illium_b_t]

    def illium_b_b_point(self):
        return self.points[illium_b_b]

    def _find_points(self):
        points = np.repeat([[0.0, 0.0]], 5, axis=0)

        red_layer = self.image_3d[:, :, 0] > 245
        green_layer = self.image_3d[:, :, 1] > 245
        blue_layer = self.image_3d[:, :, 2] > 245

        red_mask = np.logical_and(red_layer, np.logical_and(np.logical_not(green_layer), np.logical_not(blue_layer)))
        green_mask = np.logical_and(np.logical_not(red_layer), np.logical_and(green_layer, np.logical_not(blue_layer)))
        blue_mask = np.logical_and(np.logical_not(red_layer), np.logical_and(np.logical_not(green_layer), blue_layer))
        yellow_mask = np.logical_and(red_layer, np.logical_and(green_layer, np.logical_not(blue_layer)))

        red_pixels = np.array(np.argwhere(red_mask))
        mid_coord = np.mean(red_pixels[:, 1])
        l_mask = red_pixels[:, 1] < mid_coord
        l_red_pixels = red_pixels[l_mask]
        r_red_pixels = red_pixels[np.logical_not(l_mask)]
        green_pixels = np.array(np.argwhere(green_mask))
        blue_pixels = np.array(np.argwhere(blue_mask))
        yellow_pixels = np.array(np.argwhere(yellow_mask))

        points[illium_t_l] = (np.mean(l_red_pixels, axis=0))
        points[illium_t_r] = (np.mean(r_red_pixels, axis=0))
        if STRAIGHTEN_ANNOTATION_FHC:
            # make the illium line always horizontal
            points[illium_t_l, 0] = (int(points[illium_t_l, 0] + points[illium_t_r, 0]))/2
            points[illium_t_r, 0] = points[illium_t_l, 0]
        points[labrum] = (np.mean(green_pixels, axis=0))
        points[illium_b_t] = (np.mean(blue_pixels, axis=0))
        points[illium_b_b] = (np.mean(yellow_pixels, axis=0))

        return points

    def produce_point_image(self):
        output_image = np.zeros((self.image_3d.shape[0], self.image_3d.shape[1], 4), dtype=np.uint8)
        colours = [RGBA_RED, RGBA_RED, RGBA_GREEN, RGBA_YELLOW, RGBA_BLUE]

        for index in range(0, self.points.shape[0]):
            for y in range(-POINT_SIZE, POINT_SIZE):
                for x in range(-POINT_SIZE, POINT_SIZE):
                    if((y * y) + (x * x) <= POINT_SIZE*POINT_SIZE) and ((self.points[index, 0] + y) >= 0) and ((self.points[index, 1] + x) >= 0):
                        output_image[int(self.points[index, 0] + y), int(self.points[index, 1] + x)] = colours[index]

        return output_image[:, :, 0:3]

    def crop(self, shape):
        crop_shape = []
        padding = []
        for i in range(0, len(self.image_3d.shape[:2])):
            if (self.image_3d.shape[i]-shape[i]) >= 0:
                crop_shape.append(shape[i])
                padding.append(0)
            else:
                crop_shape.append(self.image_3d.shape[i])
                total_to_pad = shape[i] - self.image_3d.shape[i]
                lhs = math.floor(total_to_pad/2)
                padding.append(lhs)

        difference = np.subtract(self.image_3d.shape[:2], crop_shape)
        corner_top_left = np.int64(np.floor(difference/2))
        for i in range(0, len(self.points)):
            self.points[i] = self.points[i] - corner_top_left + padding
        self.image_3d = np.zeros(shape)

    def down_sample(self, shape):
        sy, sx = shape
        for i in range(0, len(self.points)):
            y, x = self.points[i]
            self.points[i] = ((y/sy), (x/sx))

        im_y, im_x = self.image_3d.shape[0:2]
        self.image_3d = np.zeros(shape=(int(im_y/sy), int(im_x/sx)))



