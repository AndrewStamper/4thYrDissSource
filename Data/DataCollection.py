from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import MaskCollection
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# pair of ground truth masks and their corresponding ultrasound scan
class _Scan:
    def __init__(self, scan_number, predictions_only=False):
        self.ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))
        self.predictions_only = predictions_only
        if not self.predictions_only:
            self.ground_truth = MaskCollection(scan_number)
            self.predicted_mask = None

    def crop(self, shape):
        self.ultrasound_scan.crop(shape)
        if not self.predictions_only:
            self.ground_truth.crop(shape)

    def max_pool(self, shape):
        self.ultrasound_scan = self.ultrasound_scan.down_sample(shape)
        if not self.predictions_only:
            self.ground_truth.down_sample(shape)

    def make_prediction(self, segmentation_machine_learning):
        x = np.zeros((*self.ultrasound_scan.image_3d.shape, 3))
        x[:, :, 0] = self.ultrasound_scan.image_3d
        self.predicted_mask = MaskCollection(generated_mask=segmentation_machine_learning.make_prediction(x))

    def display(self):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask', 'illium difference', 'femoral head difference', 'labrum difference']

        display_list = [self.ultrasound_scan.image_3d.reshape((*self.ultrasound_scan.image_3d.shape, 1))]
        if not self.predictions_only:
            display_list.append(self.ground_truth.as_RGB())
            if self.predicted_mask is not None:
                display_list.append(self.predicted_mask.as_RGB())
                display_list = display_list + self.ground_truth.difference_masks(self.predicted_mask)

        for i in range(len(display_list)):
            plt.subplot(2, len(display_list)//2, i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()


# each object of this type will be a set of scans and their corresponding masks
class ScanCollection:
    def __init__(self, scan_numbers, predictions_only=False):
        self.predictions_only = predictions_only
        self.scan_numbers = np.asarray(scan_numbers)
        v_scans = np.vectorize(lambda a: _Scan(a, predictions_only=predictions_only))
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

    def display(self):
        for scan in self.scans:
            scan.display()

