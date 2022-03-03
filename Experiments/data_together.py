from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *


def explore_image_segmentation():
    # load data
    # training_scan_numbers = ["A001L", "A002L",
    #                          "A021R", "A023R", "A024L", "A024R", "A026L",
    #                          "A030L", "A032L", "A032R", "A034L",
    #                          "A044R", "A045R", "A046R",
    #                          "A052R", "A053L"]
    # validation_scan_numbers = ["A001R", "A003L",
    #                            "A022R", "A030R", "A055L"]

    training_scan_numbers = []
    validation_scan_numbers = []

    for number in range(1, 40):
        for letter in ["L", "R"]:
            if number < 10:
                t_number = "0" + str(number)
            else:
                t_number = str(number)
            training_scan_numbers.append("A0" + t_number + letter)

    training_scan_numbers.remove("A005R")
    training_scan_numbers.remove("A018R")

    for number in range(40, 50):
        for letter in ["L", "R"]:
            if number < 10:
                t_number = "0" + str(number)
            else:
                t_number = str(number)
            validation_scan_numbers.append("A0" + t_number + letter)

    validation_scan_numbers.remove("A041R")

    # load data into my format
    training_data = ScanCollection(training_scan_numbers)
    validation_data = ScanCollection(validation_scan_numbers)

    # crop data
    training_data.crop(CROP_SHAPE)
    validation_data.crop(CROP_SHAPE)
    training_data.max_pool((3, 3))
    validation_data.max_pool((3, 3))

    # convert data into format for tensorflow
    train_dataset = training_data.load_data()
    validation_dataset = validation_data.load_data()

    # run segmentation model on data
    ml = ML()
    ml.configure_training_data_pipeline(train_dataset, augmentations_type=AUGMENTATION_TYPE)
    ml.configure_validation_data_pipeline(validation_dataset)
    ml.display_example()
    ml.train()
    ml.show_epoch_progression()
    ml.display_example(predictions=True, train_data=False, num=BATCH_SIZE)

    # save the model for re-use later
    ml.save_model(filename=SAVE_MODEL_AS)
