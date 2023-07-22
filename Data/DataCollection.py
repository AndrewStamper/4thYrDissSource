from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import MaskCollection
from ImageClasses.Points.AbhiAnnotationPoints import AbhiAnnotationPointScan
from FHC.MaskToFemoralPoints import *
from FHC.MaskToIlliumPoints import *
from FHC.Oracle import check_oracle_fhc
from FHC.Calculate import *
from Constants import FHC_ILLIUM_STEP, OUTPUT_FILE, OUTPUT_SEGMENTATION_FILE, RGBA_YELLOW
from FHC.Defs import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics

MASK_GROUND_TRUTH = 0
MASK_PREDICTED = 1
DDH_ORACLE = 2
GENERATED_POINTS = 3
ANNOTATION_POINTS = 4


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

    def remove_rightmost(self, x):
        self.ultrasound_scan.remove_rightmost(x)
        if not self.predictions_only:
            self.ground_truth.remove_rightmost(x)
            self.points.remove_rightmost(x)
        if self.predicted_mask is not None:
            self.predicted_mask.remove_rightmost(x)

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
        plt.savefig(OUTPUT_SEGMENTATION_FILE + str(self.scan_number))
        plt.show()

    def show_segmentation(self, display_list, title_list):
        display_list = display_list + [self.ultrasound_scan.convert_to_rgba().image_3d, self.ground_truth.as_RGB(), self.predicted_mask.as_RGB()]
        title_list = title_list + [(str(self.scan_number) + " Ultrasound Scan"), (str(self.scan_number) + " Ground Truth"), (str(self.scan_number) + " Generated Segmentation")]
        return (display_list, title_list)

    def calculate_fhc_diagnosis(self, mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=False, precise=False, use_specific_decion_value=False, specific_decion_value=0.5):
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

        return fhc_diagnosis(point_1_illium, point_2_illium, femoral_head_point_bottom, femoral_head_point_top, verbose=verbose, precise=precise, use_specific_decion_value=use_specific_decion_value, specific_decion_value=specific_decion_value)

    def calculate_fhc_percent(self, mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=False):
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

        return fhc_percentage(point_1_illium, point_2_illium, femoral_head_point_bottom, femoral_head_point_top, verbose=verbose)

    def write_mask_to_pbm(self, filename, location=OUTPUT_FILE):
        if self.predicted_mask is not None:
            self.predicted_mask.write_to_pbm("P" + filename, location)
            self.ground_truth.write_to_pbm("G" + filename, location)


# each object of this type will be a set of scans and their corresponding masks
class ScanCollection:
    def __init__(self, scan_numbers, predictions_only=False):
        self.predictions_only = predictions_only
        self.scan_numbers = np.asarray(scan_numbers)
        v_scans = np.vectorize(lambda a: SingleScan(a, predictions_only=predictions_only))
        self.scans = v_scans(self.scan_numbers)

    def number_of_scans(self):
        return self.scans.shape[0]

    def get_scan(self, scan_number):
        if scan_number in self.scan_numbers:
            index = np.where(self.scan_numbers == scan_number)
            return self.scans[index[0][0]]
        else:
            raise ValueError('Scan is not in the set' + str(scan_number))

    def crop(self, shape):
        for scan in self.scans:
            scan.crop(shape)

    def max_pool(self, shape):
        for scan in self.scans:
            scan.max_pool(shape)

    def remove_rightmost(self, x):
        for scan in self.scans:
            scan.remove_rightmost(x)

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

    def display_three_segmentations(self, scan1, scan2, scan3):
        fig = plt.figure(figsize=(15, 15))
        title_list = []
        display_list = []

        display_list, title_list = (self.get_scan(scan1)).show_segmentation(display_list, title_list)
        display_list, title_list = (self.get_scan(scan2)).show_segmentation(display_list, title_list)
        display_list, title_list = (self.get_scan(scan3)).show_segmentation(display_list, title_list)

        for i in range(len(display_list)):
            subfig = plt.subplot(3, 3, i+1)
            plt.title(title_list[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    def calculate_fhc(self, mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=False, precise=False):
        error_log = ""
        incorrect_guess = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0, len(self.scan_numbers)):
            if verbose:
                print(self.scan_numbers[i] + "-----------------------------------------------------------------------")
                print("calculated:")

            calc = self.scans[i].calculate_fhc_diagnosis(mask=mask, illium=GENERATED_POINTS, verbose=verbose, precise=precise)  # ,use_specific_decion_value=True, specific_decion_value=0.60)
            if verbose:
                print("Truth:")
            if compare_to == MASK_GROUND_TRUTH:
                true_fhc = self.scans[i].calculate_fhc_diagnosis(mask=MASK_GROUND_TRUTH, illium=ANNOTATION_POINTS, verbose=verbose, precise=precise)
            else:
                true_fhc = check_oracle_fhc(self.scan_numbers[i], precise=precise)

            if calc == FHC_NORMAL and true_fhc == FHC_NORMAL:
                tn += 1
            elif calc != FHC_NORMAL and true_fhc != FHC_NORMAL:
                tp += 1
            elif calc == FHC_NORMAL and true_fhc != FHC_NORMAL:
                fn += 1
            elif calc != FHC_NORMAL and true_fhc == FHC_NORMAL:
                fp += 1

            if calc != true_fhc:
                error_log += str(self.scan_numbers[i]) + " : " + str(calc) + " : " + str(true_fhc) + "\n"
                incorrect_guess += 1

        correct_guess = len(self.scan_numbers) - incorrect_guess
        print("Correct FHC: " + str(correct_guess) + " Incorrect FHC: " + str(incorrect_guess) + " Accuracy: " + str(correct_guess/(incorrect_guess + correct_guess)))

        print("TRUE POSITIVE : " + str(tp))
        print("TRUE NEGATIVE : " + str(tn))
        print("FALSE POSITIVE : " + str(fp))
        print("FALSE NEGATIVE  : " + str(fn))


        print("TRUE POSITIVE RATE : " + str(tp/(tp+fn)))
        print("TRUE NEGATIVE RATE : " + str(tn/(tn+fp)))
        print("FALSE POSITIVE RATE : " + str(fp/(fp+tn)))
        print("FALSE NEGATIVE RATE : " + str(fn/(fn+tp)))

        print(error_log)
        return [correct_guess/(incorrect_guess + correct_guess), tp/(tp+fn), tn/(tn+fp)]

    def calculate_fhc_csv(self):
        output = "scan, calculated, true, \n"

        for i in range(0, len(self.scan_numbers)):
            calculated = self.scans[i].calculate_fhc_percent(mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=False)* 100
            truth = self.scans[i].calculate_fhc_percent(mask=MASK_GROUND_TRUTH, illium=ANNOTATION_POINTS, verbose=False)* 100
            triple = self.scan_numbers[i] + ", " + str(calculated) + ", " + str(truth) + "\n"
            output = output + triple

        print(output)

    def calculate_fhc_scatter(self, verbose=False):
        correct_calculated_fhc_list = []
        correct_ground_truth_fhc_list = []
        errorfp_calculated_fhc_list = []
        errorfp_ground_truth_fhc_list = []
        errorfn_calculated_fhc_list = []
        errorfn_ground_truth_fhc_list = []

        for i in range(0, len(self.scan_numbers)):
            if verbose:
                print(self.scan_numbers[i] + "-----------------------------------------------------------------------")
                print("calculated:")
            calculated = self.scans[i].calculate_fhc_percent(mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=verbose)* 100
            if verbose:
                print("Truth:")
            truth = self.scans[i].calculate_fhc_percent(mask=MASK_GROUND_TRUTH, illium=ANNOTATION_POINTS, verbose=verbose)* 100

            if calculated <= 50 and truth > 50:
                errorfp_calculated_fhc_list.append(calculated)
                errorfp_ground_truth_fhc_list.append(truth)
                print("FALSE POSITIVE: " + str(self.scan_numbers[i]) + " : " + str(calculated) + " : " + str(truth) + "\n")
            elif calculated > 50 and truth <= 50:
                errorfn_calculated_fhc_list.append(calculated)
                errorfn_ground_truth_fhc_list.append(truth)
                print("FALSE NEGATIVE: " + str(self.scan_numbers[i]) + " : " + str(calculated) + " : " + str(truth) + "\n")
            else:
                correct_calculated_fhc_list.append(calculated)
                correct_ground_truth_fhc_list.append(truth)
                if abs(calculated-truth) < 0.5:
                    print("CORRECT: " + str(self.scan_numbers[i]) + " : " + str(calculated) + " : " + str(truth) + "\n")

        x = correct_calculated_fhc_list
        y = correct_ground_truth_fhc_list
        plt.scatter(x, y, alpha=0.5, color='green')

        x = errorfp_calculated_fhc_list
        y = errorfp_ground_truth_fhc_list
        plt.scatter(x, y, alpha=0.5, color='blue')

        x = errorfn_calculated_fhc_list
        y = errorfn_ground_truth_fhc_list
        plt.scatter(x, y, alpha=0.5, color='red')

        plt.xlabel('Algorithm calculated FHC%')
        plt.ylabel('Ground Truth calculated FHC%')
        plt.title('Algorithm calculated vs Ground Truth FHC scatter plot')

        # line for FHC diagnosis for algorithm
        plt.plot([50, 50], [0, 70], '--', color='black')
        # line for FHc diagnosis for ground truth
        plt.plot([0, 70], [50, 50], '--', color='black')

        plt.show()

    def calculate_roc(self, verbose=False):
        probability_of_ddh = []
        diagnosis_list = []

        for i in range(0, len(self.scan_numbers)):
            if verbose:
                print(self.scan_numbers[i] + "-----------------------------------------------------------------------")
                print("calculated:")
            calculated = self.scans[i].calculate_fhc_percent(mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=verbose)
            if verbose:
                print("Truth:")
            truth = self.scans[i].calculate_fhc_diagnosis(mask=MASK_GROUND_TRUTH, illium=ANNOTATION_POINTS, verbose=verbose)

            probability_of_ddh.append(1-calculated)
            diagnosis_list.append(truth)

        fpr, tpr, _ = metrics.roc_curve(diagnosis_list,  probability_of_ddh)

        plt.plot(fpr, tpr, color='#785ef0')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.scatter(0.118, 0.906, marker='x', color='black', label="50% Threshold", s=100)  # 50\% decision value
        plt.scatter(0.058823529411764705, 0.78125, marker='o', color='black', label="45% Threshold", s=100)  # 45\% decision value
        plt.scatter(0.4117647058823529, 0.96875, marker='^', color='black', label="54% Threshold", s=100)  # 54\% decision value
        # plt.legend(loc="lower right")
        plt.title("ROC curve produced when adjusting the algorithm's\n threshold for deciding DDH")
        plt.show()

    def calculate_fhc_diagnosis_comparison(self, verbose=False, precise=False):
        calc = np.zeros(len(self.scan_numbers))
        graphical = np.zeros(len(self.scan_numbers))
        oracle = np.zeros(len(self.scan_numbers))

        for i in range(0, len(self.scan_numbers)):
            calc[i] = self.scans[i].calculate_fhc_diagnosis(mask=MASK_PREDICTED, illium=GENERATED_POINTS, verbose=verbose, precise=precise)
            graphical[i] = self.scans[i].calculate_fhc_diagnosis(mask=MASK_GROUND_TRUTH, illium=ANNOTATION_POINTS, verbose=verbose, precise=precise)
            oracle[i] = check_oracle_fhc(self.scan_numbers[i], precise=precise)

        x = oracle
        y = graphical

        agreement_count = np.count_nonzero((x == y) == 1)
        agreement = agreement_count/len(self.scan_numbers)
        print("agreement:")
        print(agreement_count)
        print(agreement)

        random_yes = (np.count_nonzero(x == 1)/len(self.scan_numbers)) * (np.count_nonzero(y == 1)/len(self.scan_numbers))
        random_no = (1-(np.count_nonzero(x == 1)/len(self.scan_numbers))) * (1-(np.count_nonzero(y == 1)/len(self.scan_numbers)))
        random_agreement = random_yes + random_no
        print("random:")
        print(random_yes)
        print(random_no)
        print(random_agreement)

        cohens_kappa = (agreement-random_agreement)/(1-random_agreement)
        print("Cohens_Kappa:")
        print(cohens_kappa)


        print("3-way")
        c_g_agree = (calc == graphical)
        g_o_agree = (graphical == oracle)
        all_agree = np.count_nonzero(np.logical_and(c_g_agree, g_o_agree))/len(self.scan_numbers)
        print("agreement")
        print(all_agree)


        ratings = np.append(np.append([oracle], [calc], axis=0), [graphical], axis=0).transpose()
        print(ratings.shape)

        N, R = ratings.shape
        NR = N * R
        categories = set(ratings.ravel().tolist())
        P_example = -np.full(N, R)
        p_class = 0.0
        for c in categories:
            c_sum = np.sum(ratings == c, axis=1)
            P_example += c_sum**2
            p_class += (np.sum(c_sum) / float(NR)) ** 2
        P_example = np.sum(P_example) / float(NR * (R-1))
        k = (P_example - p_class) / (1 - p_class)
        print("Fleiss' kappa = {:.3f}".format(k))

