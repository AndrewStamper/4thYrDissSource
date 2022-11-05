from FHC.Defs import *


def fhc_percentage(illium_point_l, illium_point_r, femoral_head_point_bottom, femoral_head_point_top, verbose=False):
    y1, x1 = illium_point_l
    y2, x2 = illium_point_r
    y3, x3 = femoral_head_point_bottom
    y4, x4 = femoral_head_point_top
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

    diff_below = y3 - py
    diff = y3 - y4
    coverage = diff_below / diff

    if verbose:
        print("i left: " + str(illium_point_l) + " i right: " + str(illium_point_r))
        print("fh top: " + str(femoral_head_point_top) + " fh bottom: " + str(femoral_head_point_bottom))
        print("intersection: " + str((px, py)))
        print("diff_below: " + str(diff_below))
        print("diff: " + str(diff))
        print("fhc coverage: " + str(coverage))

    return coverage


def fhc_diagnosis(illium_point_l, illium_point_r, femoral_head_point_bottom, femoral_head_point_top, verbose=False, precise=False, use_specific_decion_value=False, specific_decion_value=0.5):
    coverage = fhc_percentage(illium_point_l, illium_point_r, femoral_head_point_bottom, femoral_head_point_top, verbose)

    if use_specific_decion_value:
        decision_value = specific_decion_value
    else:
        decision_value = 0.5

    if coverage > decision_value:
        result = FHC_NORMAL
    else:
        if precise:
            if coverage > 0.4:
                result = FHC_DYSPLASTIC
            else:
                result = FHC_DECENTERD
        else:
            result = FHC_ABNORMAL

    if verbose:
        print("result: " + str(result))

    return result

