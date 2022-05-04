from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *
from Segmentation.Augmentations.Augmentation_Constants import *
import time


def train_all_models():
    augmentation_types = [AUGMENTATION_TYPE_NONE, AUGMENTATION_TYPE_BASIC, AUGMENTATION_TYPE_MEDIUM]
    model_types = [MODEL_UNET, MODEL_ENCODER_MOD_UNET, MODEL_ENCODER_DECODER_MODIFIED_UNET]
    for augmentation_type in augmentation_types:
        for model_type in model_types:
            if (model_type == MODEL_ENCODER_DECODER_MODIFIED_UNET and augmentation_type == AUGMENTATION_TYPE_MEDIUM) or \
                    (model_type == MODEL_ENCODER_MOD_UNET and augmentation_type == AUGMENTATION_TYPE_MEDIUM):
                continue

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
            ml = ML(model_type=model_type)
            ml.configure_training_data_pipeline(train_dataset, augmentations_type=augmentation_type)
            ml.configure_validation_data_pipeline(validation_dataset)
            start = time.time()
            ml.train()
            time_elapsed = time.time() - start

            # save the model for re-use later
            if model_type == MODEL_ORIGINAL_MODIFIED:
                model_type_text = "testing_model"
                model_type_name = "testing U-Net"
            elif model_type == MODEL_UNET:
                model_type_text = "unet"
                model_type_name = "U-Net"
            elif model_type == MODEL_ENCODER_MOD_UNET:
                model_type_text = "encoderonly"
                model_type_name = "Encoder modified U-Net"
            else:  # model_type== MODEL_ENCODER_DECODER_MODIFIED_UNET:
                model_type_text = "encoderdecoder"
                model_type_name = "Encoder and Decoder modified U-Net"

            if augmentation_type == AUGMENTATION_TYPE_NONE:
                augmentation_type_text = "no"
            elif augmentation_type == AUGMENTATION_TYPE_BASIC:
                augmentation_type_text = "type 1"
            else:  # AUGMENTATION_TYPE_MEDIUM:
                augmentation_type_text = "type 2"

            model_name = model_type_text+augmentation_type_text

            ml.show_epoch_progression(model_type_name, augmentation_type_text)

            ml.save_model(filename=model_name)

            x = np.array([time_elapsed])
            np.savetxt(OUTPUT_FILE + model_name + ".txt", x, delimiter=',', fmt='%f')
            print("completed for " + model_name)

