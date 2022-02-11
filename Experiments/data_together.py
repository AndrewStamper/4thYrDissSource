from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *


def test_data():
    # load data
    training_scan_numbers = ["A001L", "A002L",
                             "A021R", "A023R", "A024L", "A024R", "A026L",
                             "A030L", "A032L", "A032R", "A034L",
                             "A044R", "A045R", "A046R",
                             "A052R", "A053L"]
    validation_scan_numbers = ["A001R", "A003L",
                               "A022R", "A030R", "A055L"]

    # load data into my format
    training_data = ScanCollection(training_scan_numbers)
    validation_data = ScanCollection(validation_scan_numbers)

    # crop data
    training_data.crop(INPUT_SHAPE)
    validation_data.crop(INPUT_SHAPE)

    # convert data into format for tensorflow
    train_dataset = training_data.load_data()
    validation_dataset = validation_data.load_data()

    # run segmentation model on data
    ml = ML(train_dataset, validation_dataset)
    ml.display_example()
    ml.train()
    ml.show_epoch_progression()
    ml.display_example(predictions=True, train_data=False, num=3)
