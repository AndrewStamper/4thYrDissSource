import numpy as np
from queue import Queue


def find_extrema(mask):
    femoral_head_mask = mask.femoral_head.image_3d
    mask_pixels = np.array(np.argwhere(femoral_head_mask))
    femoral_head_point_bottom = (mask_pixels[np.argmax(mask_pixels, axis=0)[0]])  # bottom is max because origin is top left
    femoral_head_point_top = (mask_pixels[np.argmin(mask_pixels, axis=0)[0]])  # top is min symmetrically
    return femoral_head_point_bottom, femoral_head_point_top


def walk_to_extrema(mask):
    femoral_head_mask = mask.femoral_head.image_3d
    mask_pixels = np.array(np.argwhere(femoral_head_mask))
    mask_center = np.mean(mask_pixels, axis=0, dtype=int)

    if femoral_head_mask[mask_center[0], mask_center[1]] != 255.0:
        print("mean point is not part of mask, default to finding extrema, Note this may affect accuracy")
        return find_extrema(mask)
    else:
        femoral_head_point_bottom, femoral_head_point_top = _walk_to_extrema(femoral_head_mask, mask_center)

    return femoral_head_point_bottom, femoral_head_point_top


def _walk_to_extrema(femoral_head_mask, start):
    lowest = start
    highest = start
    q = Queue()
    q.put((start[0], start[1]))
    v = {(start[0], start[1])}
    while not q.empty():
        loc = q.get()
        if loc[0] > lowest[0]:
            lowest = loc
        if loc[0] < highest[0]:
            highest = loc
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if not (dy == 0 and dx == 0):
                    new_loc = (loc[0] + dy, loc[1] + dx)
                    if (new_loc[0] < femoral_head_mask.shape[0] and
                        new_loc[1] < femoral_head_mask.shape[1] and
                            femoral_head_mask[new_loc] == 255.0 and
                            not(new_loc in v)):
                        q.put(new_loc)
                        v.add(new_loc)

    return lowest, highest
