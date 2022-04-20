from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import MaskCollection
from ImageClasses.Points.AbhiAnnotationPoints import AbhiAnnotationPointScan
from FHC.MaskToFemoralPoints import *
from FHC.MaskToIlliumPoints import *
from FHC.Oracle import check_oracle_fhc
from FHC.Calculate import *
from Constants import FHC_ILLIUM_STEP
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MASK_GROUND_TRUTH = 0
MASK_PREDICTED = 1
DDH_ORACLE = 2

GENERATED_POINTS = 0
ANNOTATION_POINTS = 1


# masks and their corresponding ultrasound scan
class SingleScan:
    def __init__(self, scan_number, predictions_only=False):
        self.scan_number = scan_number
        self.ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))
        self.predictions_only = predictions_only
        self.predicted_mask = None
        if not self.predictions_only:
            self.points = AbhiAnnotationPointScan(filename=(scan_number + "_g.png"))
            self.ground_truth = MaskCollection(scan_number)

    def crop(self, shape):
        self.ultrasound_scan.crop(shape)
        if not self.predictions_only:
            self.ground_truth.crop(shape)
            self.points.crop(shape)
        if self.predicted_mask is not None:
            self.predicted_mask.crop(shape)

    def max_pool(self, shape):
        self.ultrasound_scan = self.ultrasound_scan.down_sample(shape)
        if not self.predictions_only:
            self.ground_truth.down_sample(shape)
            self.points.down_sample(shape)
        if self.predicted_mask is not None:
            self.predicted_mask.down_sample(shape)

    def make_prediction(self, segmentation_machine_learning):
        x = np.zeros((*self.ultrasound_scan.image_3d.shape, 3))
        x[:, :, 0] = self.ultrasound_scan.image_3d
        self.predicted_mask = MaskCollection(generated_mask=segmentation_machine_learning.make_prediction(x))

    def display(self):
        fig = plt.figure(figsize=(15, 15))
        title_list = [(str(self.scan_number) + " Ultrasound Scan")]

        display_list = [self.ultrasound_scan.image_3d.reshape((*self.ultrasound_scan.image_3d.shape, 1))]
        if not self.predictions_only:
            display_list = display_list + [self.ground_truth.as_RGB(), self.points.produce_point_image()]
            title_list = title_list + ["Ground Truth Mask", "Ground Truth Points"]
        if self.predicted_mask is not None:
            display_list.append(self.predicted_mask.as_RGB())
            title_list.append("Algorithm Generated Mask")
            display_list.append(find_upper_with_stopping(self.predicted_mask, FHC_ILLIUM_STEP))
            title_list.append("Upper Illium")
        if (self.predicted_mask is not None) and (not self.predictions_only):
            display_list = display_list + self.ground_truth.difference_masks(self.predicted_mask)
            title_list = title_list + ['Illium difference', 'Femoral Head difference', 'Labrum difference']

        for i in range(len(display_list)):
            subfig = plt.subplot(3, 3, i+1)
            plt.title(title_list[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    def calculate_fhc(self, mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=False, precise=False):
        if mask is MASK_GROUND_TRUTH:
            mask = self.ground_truth
        elif mask is MASK_PREDICTED:
            assert(self.predicted_mask is not None)
            mask = self.predicted_mask
        else:
            print("mask type unsupported, defaulting to predicted mask")
            assert(self.predicted_mask is not None)
            mask = self.predicted_mask

        femoral_head_point_bottom, femoral_head_point_top = walk_to_extrema(mask)

        if illium == GENERATED_POINTS:
            point_1_illium, point_2_illium = illium_points(mask, FHC_ILLIUM_STEP)
        else:
            point_1_illium = self.points.illium_t_l_point()
            point_2_illium = self.points.illium_t_r_point()

        return fhc(point_1_illium, point_2_illium, femoral_head_point_bottom, femoral_head_point_top, verbose=verbose, precise=precise)


# each object of this type will be a set of scans and their corresponding masks
class ScanCollection:
    def __init__(self, scan_numbers, predictions_only=False):
        self.predictions_only = predictions_only
        self.scan_numbers = np.asarray(scan_numbers)
        v_scans = np.vectorize(lambda a: SingleScan(a, predictions_only=predictions_only))
        self.scans = v_scans(self.scan_numbers)

    def number_of_scans(self):
        return self.scans.shape(0)

    def get_scan(self, scan_number):
        if scan_number in self.scan_numbers:
            index = np.where(self.scan_numbers == scan_number)
            return self.scans[index[0][0]]
        else:
            raise ValueError('Scan is not in the set')

    def crop(self, shape):
        for scan in self.scans:
            scan.crop(shape)

    def max_pool(self, shape):
        for scan in self.scans:
            scan.max_pool(shape)

    def load_data(self):
        x = np.zeros((*self.scans.shape, *self.scans[0].ultrasound_scan.image_3d.shape, 3))
        y = np.zeros((*self.scans.shape, *self.scans[0].ultrasound_scan.image_3d.shape, 1))

        for i in range(0, self.scans.shape[0]):
            x[i, :, :, 0] = self.scans[i].ultrasound_scan.image_3d
            if not self.predictions_only:
                y[i, :, :] = self.scans[i].ground_truth.as_segmentation_mask()

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        return dataset

    def make_predictions(self, segmentation_machine_learning):
        for scan in self.scans:
            scan.make_prediction(segmentation_machine_learning)

    def display(self, display_list=None):
        if display_list is None:
            for scan in self.scans:
                scan.display()
        else:
            for elem in display_list:
                num_list = np.argwhere(self.scan_numbers == elem)
                if num_list.size == 0:
                    print("Scan number : " + str(elem) + " is not found in this scan collection, cannot print")
                else:
                    self.scans[num_list[0][0]].display()

    def calculate_fhc(self, mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=False, precise=False):
        error_log = ""
        incorrect_guess = 0
        for i in range(0, len(self.scan_numbers)):
            calc = self.scans[i].calculate_fhc(mask=mask, verbose=verbose, precise=precise)

            if compare_to == MASK_GROUND_TRUTH:
                true_fhc = self.scans[i].calculate_fhc(mask=MASK_GROUND_TRUTH, verbose=verbose, precise=precise)
            else:
                true_fhc = check_oracle_fhc(self.scan_numbers[i], precise=precise)

            if calc != true_fhc:
                error_log += str(self.scan_numbers[i]) + " : " + str(calc) + " : " + str(true_fhc) + "\n"
                incorrect_guess += 1

        correct_guess = len(self.scan_numbers) - incorrect_guess
        print("Correct FHC: " + str(correct_guess) + " Incorrect FHC: " + str(incorrect_guess) + " Accuracy: " + str(correct_guess/(incorrect_guess + correct_guess)))
        print(error_log)

