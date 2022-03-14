from Data.DataCollection import SingleScan, MASK_GROUND_TRUTH
from FHC.Oracle import check_oracle_fhc
from Data.DataCollection import ScanCollection, MASK_GROUND_TRUTH, MASK_PREDICTED
from Segmentation.Interface import ML
from Constants import *
from Experiments.DataSplits import *


def explore_FHC(scan_number):
    scan = SingleScan(scan_number)
    calc = scan.calculate_fhc(mask=MASK_GROUND_TRUTH, verbose=False, precise=False)
    oracle = check_oracle_fhc(scan_number, precise=False)
    print("calculated: " + str(calc) + " oracle: " + str(oracle))

    # load data into my format
    validation_data = ScanCollection(validation_data_list())

    # crop data
    validation_data.crop(CROP_SHAPE)
    validation_data.max_pool((3, 3))

    # convert data into format for tensorflow
    validation_dataset = validation_data.load_data()

    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(validation_dataset)
    new_ml.load_model(filename=SAVED_MODEL)

    # check it has been loaded correctly
    new_ml.evaluate_model()

    # make predictions and calculate FHC
    validation_data.make_predictions(new_ml)
    validation_data.calculate_fhc(mask=MASK_PREDICTED, verbose=False, precise=False)

    validation_data.display(["A042R", "A043L", "A045R", "A047L"])

