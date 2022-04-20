from Data.DataCollection import SingleScan, ScanCollection, MASK_GROUND_TRUTH, MASK_PREDICTED
from FHC.Oracle import check_oracle_fhc
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *


def explore_FHC(scan_number):
    print(scan_number)
    scan = SingleScan(scan_number)
    calc = scan.calculate_fhc(mask=MASK_GROUND_TRUTH, verbose=True, precise=False)
    oracle = check_oracle_fhc(scan_number, precise=False)
    print("calculated: " + str(calc) + " oracle: " + str(oracle))

    # load data into my format
    validation_data = ScanCollection(validation_data_list())

    # crop data
    validation_data.crop(CROP_SHAPE)
    validation_data.max_pool(DOWN_SAMPLE_SHAPE)

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

    print("FHC using predicted masks")
    validation_data.calculate_fhc(mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=False, precise=False)

    validation_data.display()# ["A072R", "A079L", "A045R"]

