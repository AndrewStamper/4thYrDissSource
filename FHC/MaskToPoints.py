import numpy as np


def find_extrema(mask):
    femoral_head_mask = mask.femoral_head.image_3d
    mask_pixels = np.array(np.argwhere(femoral_head_mask))
    femoral_head_point_bottom = (mask_pixels[np.argmax(mask_pixels, axis=0)[0]])  # bottom is max because origin is top left
    femoral_head_point_top = (mask_pixels[np.argmin(mask_pixels, axis=0)[0]])  # top is min symmetrically
    return femoral_head_point_bottom, femoral_head_point_top
