from Data.DataCollection import SingleScan, ScanCollection, MASK_GROUND_TRUTH, MASK_PREDICTED
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *


# analysis of the scans which were deemed unacceptable by the medical professionals is still possible, how does the algorithm perfrom on these 'hard' examples
def unacceptable_analysis():
    # load data
    data = ScanCollection(unnaceptable_data_list(), predictions_only=True)

    # crop data
    data.crop(CROP_SHAPE)
    data.max_pool(DOWN_SAMPLE_SHAPE)

    # convert data into format for tensorflow
    dataset = data.load_data()

    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(dataset)
    new_ml.load_model(filename=SAVED_MODEL)

    # check it has been loaded correctly
    new_ml.evaluate_model()

    # make predictions and calculate FHC
    data.make_predictions(new_ml)

    print("using predicted masks")
    #data.calculate_fhc(mask=MASK_PREDICTED, verbose=False, precise=False)

    data.display()

