from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *


def explore_reloading_models():
    validation_scan_numbers = []

    for number in range(40, 50):
        for letter in ["L","R"]:
            if number < 10:
                tnumber = "0" + str(number)
            else:
                tnumber = str(number)
            validation_scan_numbers.append("A0" + tnumber + letter)

    validation_scan_numbers.remove("A041R")

    # load data into my format
    validation_data = ScanCollection(validation_scan_numbers)

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
