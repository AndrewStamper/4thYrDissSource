from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Masks.AnnotationsMask import MaskCollection
import numpy as np


# pair of ground truth masks and their corresponding ultrasound scan
class _Scan:
    def __init__(self, scan_number):
        self.ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))
        self.ground_truth = MaskCollection(scan_number)


# each object of this type will be a set of scans and their corresponding scans
class ScanCollection:
    def __init__(self, scan_numbers):
        self.scan_numbers = np.asarray(scan_numbers)
        v_scans = np.vectorize(_Scan)
        self.scans = v_scans(self.scan_numbers)

    def number_of_scans(self):
        return self.scans.shape(0)

    def get_scan(self, scan_number):
        if scan_number in self.scan_numbers:
            index = np.where(self.scan_numbers == scan_number)
            return self.scans[index[0][0]]
        else:
            raise ValueError('Scan is not in the set')
