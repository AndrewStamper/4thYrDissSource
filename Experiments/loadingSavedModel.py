from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *


def explore_reloading_models():
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

    # make predictions
    validation_data.make_predictions(new_ml)
    validation_data.display()
