from Constants import RGBA_RED, RGBA_BLUE, RGBA_GREEN, CROP_SHAPE, DOWN_SAMPLE_SHAPE, SAVED_MODEL, RGB_RED, RGB_BLUE, RGB_GREEN, RGBA_YELLOW
from Data.DataCollection import SingleScan, ScanCollection
from FHC.MaskToFemoralPoints import walk_to_extrema, find_extrema
from FHC.MaskToIlliumPoints import find_upper, find_upper_with_stopping, illium_points
from ImageClasses.Masks.AnnotationsMask import MaskCollection
from ImageClasses.NumpyImage import NumpyImage
from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
import numpy as np

from Segmentation.Interface import ML


def midpoint(y, m, image, colour):
    for x in range(0, image.shape[1]):
        for a in range(-2, 2):
            coord = round(y+(m*x)+a)
            if coord < image.shape[0]:
                image[coord, x] = colour


def printFHCdiagrams(scan_number):
    file_name = 'visualise_FHC_'
    file_number = 0

    # load data into my format
    data = ScanCollection([scan_number])

    # crop data
    data.crop(CROP_SHAPE)
    data.max_pool(DOWN_SAMPLE_SHAPE)

    # convert data into format for tensorflow
    dataset = data.load_data()

    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(dataset)
    new_ml.load_model(filename=SAVED_MODEL)

    # make predictions
    data.make_predictions(new_ml)

    # load structures
    ultrasound_scan = data.scans[0].ultrasound_scan  # UltrasoundScan(filename=(scan_number + ".jpg"))
    mask_collection = data.scans[0].predicted_mask  # MaskCollection(scan_number)

    # image itself
    ultrasound_scan.write_image(file_name + str(file_number) + '_original')
    file_number = file_number + 1
    size_of_circle = 3

    # femoral head mask
    femoral_head = mask_collection.femoral_head.convert_to_rgba(colour=RGBA_BLUE)
    annotated_femoral = ultrasound_scan.add_annotations(annotations=femoral_head)
    annotated_femoral.write_image(file_name + str(file_number) + '_femoral_head')
    file_number = file_number + 1

    # femoral head mask EXTREMA
    femoral_head = mask_collection.femoral_head.convert_to_rgba(colour=RGBA_BLUE)
    femoral_head_point_bottom, femoral_head_point_top = find_extrema(mask_collection)
    for (i, j) in [femoral_head_point_bottom, femoral_head_point_top]:
        for x in range(-size_of_circle, size_of_circle):
            for y in range(-size_of_circle, size_of_circle):
                if x*x+y*y <= size_of_circle*size_of_circle:
                    femoral_head.image_3d[i+x, j+y] = RGBA_GREEN
    annotated_femoral = ultrasound_scan.add_annotations(annotations=femoral_head)
    annotated_femoral.write_image(file_name + str(file_number) + '_femoral_head_extrema')
    file_number = file_number + 1

    # femoral head mask WALK
    femoral_head = mask_collection.femoral_head.convert_to_rgba(colour=RGBA_BLUE)
    femoral_head_point_bottom, femoral_head_point_top = walk_to_extrema(mask_collection)
    mask_pixels = np.array(np.argwhere(mask_collection.femoral_head.image_3d))
    (i, j) = np.mean(mask_pixels, axis=0, dtype=int)
    for x in range(-size_of_circle, size_of_circle):
        for y in range(-size_of_circle, size_of_circle):
            if x*x+y*y <= size_of_circle*size_of_circle:
                femoral_head.image_3d[i+x, j+y] = RGBA_RED
    for (i, j) in [femoral_head_point_bottom, femoral_head_point_top]:
        for x in range(-size_of_circle, size_of_circle):
            for y in range(-size_of_circle, size_of_circle):
                if x*x+y*y <= size_of_circle*size_of_circle:
                    femoral_head.image_3d[i+x, j+y] = RGBA_GREEN
    annotated_femoral = ultrasound_scan.add_annotations(annotations=femoral_head)
    annotated_femoral.write_image(file_name + str(file_number) + '_femoral_head_walk')
    file_number = file_number + 1

    # ilium
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium')
    file_number = file_number + 1

    # ilium stop
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    pixels = np.array(np.argwhere(find_upper_with_stopping(mask_collection, 5) == 255)[:, :2])
    for (i, j) in pixels:
        for x in range(-2, 2):
            illium.image_3d[i+x, j] = RGBA_BLUE
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium_with_stopping')
    file_number = file_number + 1

    # ilium no stop
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    pixels = np.array(np.argwhere(find_upper(mask_collection) == 255)[:, :2])
    for (i, j) in pixels:
        for x in range(-2, 2):
            illium.image_3d[i+x, j] = RGBA_BLUE
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium_no_stop')
    file_number = file_number + 1

    # ilium straight line
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    point1, point2 = illium_points(mask_collection, 5, fixed_horizontal=True)
    c = point1[0]
    m = point2[0] - point1[0]
    midpoint(c, m, illium.image_3d, RGBA_GREEN)
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium_horizontal')
    file_number = file_number + 1

    # ilium fit line
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    point1, point2 = illium_points(mask_collection, 5, fixed_horizontal=False)
    c = point1[0]
    m = point2[0] - point1[0]
    midpoint(c, m, illium.image_3d, RGBA_GREEN)
    ultrasound_scan.add_annotations(annotations=illium).write_image(file_name + str(file_number) + '_illium_best_fit')


def printISBIdiagrams(scan_number, fileModifier=""):
    file_name = fileModifier + "ISBI_"
    file_number = 0
    annotations_shape = [128, 128, 4]

    # load data into my format
    data = ScanCollection([scan_number])
    # crop data
    data.crop(CROP_SHAPE)
    data.max_pool(DOWN_SAMPLE_SHAPE)
    # convert data into format for tensorflow
    dataset = data.load_data()
    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(dataset)
    new_ml.load_model(filename=SAVED_MODEL)
    # make predictions
    data.make_predictions(new_ml)

    # load structures
    ultrasound_scan = data.scans[0].ultrasound_scan  # UltrasoundScan(filename=(scan_number + ".jpg"))
    mask_collection = data.scans[0].predicted_mask  # MaskCollection(scan_number)

    # image itself
    ultrasound_scan.write_image(file_name + str(file_number) + '_original')
    file_number = file_number + 1
    size_of_circle = 3

    # ground truth
    ground_truth = NumpyImage(rows=data.scans[0].ground_truth.as_RGB())
    ground_truth.write_image(file_name + str(file_number) + '_ground_truth')
    file_number = file_number + 1

    # generated mask
    generated_mask = NumpyImage(rows=data.scans[0].predicted_mask.as_RGB())
    generated_mask.write_image(file_name + str(file_number) + '_generated_mask')
    file_number = file_number + 1

    # femoral head mask
    femoral_head = mask_collection.femoral_head.convert_to_rgb(colour=RGB_BLUE)
    femoral_head.write_image(file_name + str(file_number) + '_femoral_head')
    file_number = file_number + 1

    # femoral head mask WALK
    femoral_head = mask_collection.femoral_head.convert_to_rgba(colour=RGBA_BLUE)
    # central dot
    mask_pixels = np.array(np.argwhere(mask_collection.femoral_head.image_3d))
    (i, j) = np.mean(mask_pixels, axis=0, dtype=int)
    for x in range(-size_of_circle, size_of_circle):
        for y in range(-size_of_circle, size_of_circle):
            if x*x+y*y <= size_of_circle*size_of_circle:
                femoral_head.image_3d[i+x, j+y] = RGBA_RED
    # extrema
    femoral_head_point_bottom, femoral_head_point_top = walk_to_extrema(mask_collection)
    extrema = NumpyImage(np.zeros(annotations_shape))
    for (i, j) in [femoral_head_point_bottom, femoral_head_point_top]:
        for x in range(-size_of_circle, size_of_circle):
            for y in range(-size_of_circle, size_of_circle):
                if x*x+y*y <= size_of_circle*size_of_circle:
                    extrema.image_3d[i+x, j+y] = RGBA_GREEN
                    femoral_head.image_3d[i+x, j+y] = RGBA_GREEN
    femoral_head = femoral_head.rgba_to_rgb()
    femoral_head.write_image(file_name + str(file_number) + '_femoral_head_walk')
    file_number = file_number + 1

    # ilium
    illium = mask_collection.illium.convert_to_rgb(colour=RGB_RED)
    illium.write_image(file_name + str(file_number) + '_illium')
    file_number = file_number + 1

    # ilium stop
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    pixels = np.array(np.argwhere(find_upper_with_stopping(mask_collection, 5) == 255)[:, :2])
    for (i, j) in pixels:
        for x in range(-2, 2):
            illium.image_3d[i+x, j] = RGBA_BLUE
    illium = illium.rgba_to_rgb()
    illium.write_image(file_name + str(file_number) + '_illium_with_stopping')
    file_number = file_number + 1

    # ilium straight line1
    illium = mask_collection.illium.convert_to_rgba(colour=RGBA_RED)
    point1, point2 = illium_points(mask_collection, 5, fixed_horizontal=True)
    c = point1[0]
    m = point2[0] - point1[0]
    midpoint(c, m, illium.image_3d, RGBA_YELLOW)
    illium = illium.rgba_to_rgb()
    illium.write_image(file_name + str(file_number) + '_illium_horizontal1')
    file_number = file_number + 1

    # ilium straight line2
    line = NumpyImage(np.zeros(annotations_shape))
    point1, point2 = illium_points(mask_collection, 5, fixed_horizontal=True)
    c = point1[0]
    m = point2[0] - point1[0]
    midpoint(c, m, line.image_3d, RGBA_YELLOW)
    fhc_anno = line + extrema
    ultrasound_scan.add_annotations(annotations=fhc_anno).write_image(file_name + str(file_number) + '_illium_horizontal2')
    file_number = file_number + 1

    # A001R - IMAGES
    demo = SingleScan("A001R")
    annotations = demo.ground_truth.as_RGBA()
    demo.ultrasound_scan.add_annotations(annotations=annotations).write_image(file_name + str(file_number) + '_A001R_ground_truth')
    file_number = file_number + 1


