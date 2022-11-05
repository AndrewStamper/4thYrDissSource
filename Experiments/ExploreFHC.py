from Data.DataCollection import SingleScan, ScanCollection, MASK_GROUND_TRUTH, MASK_PREDICTED, DDH_ORACLE, ANNOTATION_POINTS, GENERATED_POINTS
from FHC.Oracle import check_oracle_fhc
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *


def explore_FHC(scan_number):
    # load data into my format
    validation_data = ScanCollection(validation_data_list())  # validation_data_list training_data_list() + validation_data_list() test_data_list

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
    validation_data.calculate_fhc(mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=False, precise=False)  # DDH_ORACLE MASK_GROUND_TRUTH

    # validation_data.display(["A082R", "A084R", "A088R", "A092R", "A094R", "A100L", "A086R", "A099L"])

    #
    # #ALL WRONG PREDCITIONS
    # wrong_list = ["A051R", "A045R", "A074L", "A047L", "A065R"]
    # data = ScanCollection(wrong_list)
    #
    # # crop data
    # data.crop(CROP_SHAPE)
    # data.max_pool(DOWN_SAMPLE_SHAPE)
    #
    # # make predictions and calculate FHC
    # data.make_predictions(new_ml)
    #
    # print("FHC using predicted masks")
    # data.calculate_fhc(mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=True, precise=False)
    # data.display()

