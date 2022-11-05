import numpy as np
import math
from Constants import FHC_FIXED_HORIZONTAL


def illium_points(mask, step_size, fixed_horizontal=FHC_FIXED_HORIZONTAL):
    upper = find_upper_with_stopping(mask, step_size)

    if fixed_horizontal:
        # if draw a straight horizontal line at that height
        c = average_height(upper)
        m = 0
    else:
        # if fit a line using least squares
        mask_pixels = np.array(np.argwhere(upper == 255))
        m, c = calculate_line(mask_pixels[:, 0], mask_pixels[:, 1])

    return (c, 0), (c+m, 1)


def average_height(array):
    mask_pixels = np.array(np.argwhere(array))[:, 0]
    return np.average(mask_pixels)


def find_upper(mask):
    illium_mask = mask.illium.image_3d
    shape = illium_mask.shape
    output = np.zeros((*shape, 1))
    top_so_far_y = []
    top_so_far_x = []
    for x in range(0, shape[1]):
        for y in range(1, shape[0]):
            if illium_mask[y, x] > 0 and illium_mask[y-1, x] == 0:
                top_so_far_y.append(y)
                top_so_far_x.append(x)

    index = 0
    while index < len(top_so_far_x):
        output[top_so_far_y[index], top_so_far_x[index], 0] = 255
        index = index+1

    return output


def find_upper_with_stopping(mask, step_size):
    illium_mask = mask.illium.image_3d
    shape = illium_mask.shape
    output = np.zeros((*shape, 1))
    this_section_y = []
    this_section_x = []
    top_so_far_y = []
    top_so_far_x = []
    max_angle = 0
    max_angle_x = 0
    for x in range(0, shape[1]):
        for y in range(1, shape[0]):
            if illium_mask[y, x] > 0 and illium_mask[y-1, x] == 0:
                this_section_y.append(y)
                this_section_x.append(x)
        if x % step_size == 0 and x != 0 and (len(this_section_y) > step_size*0.5 or len(top_so_far_y)==0):
            if x == step_size:
                top_so_far_y = this_section_y
                top_so_far_x = this_section_x
                this_section_y = []
                this_section_x = []
            else:
                m_so_far, c_so_far = calculate_line(top_so_far_y[-3*step_size:], top_so_far_x[-3*step_size:])
                m_section, c_section = calculate_line(this_section_y, this_section_x)
                angle = compare_lines(m_so_far, m_section)
                if angle < max_angle:
                    max_angle = angle
                    max_angle_x = x

                top_so_far_y = top_so_far_y + this_section_y
                top_so_far_x = top_so_far_x + this_section_x
                this_section_y = []
                this_section_x = []
        elif x % step_size == 0 and x != 0 and len(top_so_far_y) > 2*step_size:
            break

    if len(top_so_far_x) > 5*step_size:
        end = 2*step_size
    elif len(top_so_far_x) <= 2*step_size:
        end = 0
    else:
        end = step_size

    done = False
    index = 0
    while (not done) and index < len(top_so_far_x):
        output[top_so_far_y[index], top_so_far_x[index], 0] = 255
        done = (index + end > max_angle_x)
        index = index+1

    return output


def calculate_line(y, x):
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def compare_lines(m1, m2):
    if m1*m2 != -1:
        angle = math.degrees(math.atan((m1 - m2)/(1+m1*m2)))
    else:
        if m1 - m2 > 0:
            angle = 90
        else:
            angle = -90
    return angle
