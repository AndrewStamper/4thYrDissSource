from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import AnnotationMaskScan
from Constants import *


def visualise_mask(scan_number):
    file_name = 'visualise_mask_'
    file_number = 0

    ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))

    illium = AnnotationMaskScan(filename=(scan_number + "_i.png"), colour=RGBA_RED)
    femoral_head = AnnotationMaskScan(filename=(scan_number + "_f.png"), colour=RGBA_BLUE)
    labrum = AnnotationMaskScan(filename=(scan_number + "_l.png"), colour=RGBA_GREEN)

    i_f_l = illium.combine_masks(femoral_head).combine_masks(labrum)

    ultrasound_scan.write_image(file_name + str(file_number) + '_original')
    file_number = file_number + 1
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium')
    file_number = file_number + 1
    ultrasound_scan.add_annotations(annotations=femoral_head).write_image(file_name + str(file_number) + '_femoral_head')
    file_number = file_number + 1
    ultrasound_scan.add_annotations(annotations=labrum).write_image(file_name + str(file_number) + '_labrum')
    file_number = file_number + 1
    ultrasound_scan.add_annotations(annotations=i_f_l).write_image(file_name + str(file_number) + '_all')
