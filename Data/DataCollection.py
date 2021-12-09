from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import MaskCollection
import numpy as np


# pair of ground truth masks and their corresponding ultrasound scan
class _Scan:
    def __init__(self, scan_number, predictions_only=False):
        self.ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))
        self.predictions_only = predictions_only
        if not self.predictions_only:
            self.ground_truth = MaskCollection(scan_number)

    def crop(self, shape):
        self.ultrasound_scan.crop(shape)
        if not self.predictions_only:
            self.ground_truth.crop(shape)


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

    def load_data(self):
        x = np.zeros((*self.scans.shape, *self.scans[0].ultrasound_scan.image_3d.shape))
        y = np.zeros((*x.shape, 3))

        for i in range(0, self.scans.shape[0]):
            x[i, :, :] = self.scans[i].ultrasound_scan.image_3d
            if not self.predictions_only:
                y[i, :, :] = self.scans[i].ground_truth.as_single()
        return x, y
