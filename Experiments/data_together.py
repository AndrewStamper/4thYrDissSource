from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *


def explore_image_segmentation():

    # load data into my format
    training_data = ScanCollection(training_data_list())
    validation_data = ScanCollection(validation_data_list())

    # crop data
    training_data.crop(TRAINING_CROP_SHAPE)
    validation_data.crop(CROP_SHAPE)
    training_data.max_pool(DOWN_SAMPLE_SHAPE)
    validation_data.max_pool(DOWN_SAMPLE_SHAPE)

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
    # ml.save_model(filename=SAVE_MODEL_AS)
