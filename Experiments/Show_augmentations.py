from Segmentation.Augmentations.Augmentations import *
from Data.DataCollection import *
from Constants import *
from Segmentation.Interface import ML
from PIL import Image


def show_augmentations(scan_number):
    # load data into my format
    training_data = ScanCollection([scan_number])
    validation_data = ScanCollection([scan_number])

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
    ml.configure_training_data_pipeline(train_dataset, augmentations_type=AUGMENTATION_TYPE_MEDIUM)
    ml.configure_validation_data_pipeline(validation_dataset)

    pre = validation_data.scans[0].ultrasound_scan.image_3d
    pre = pre.reshape((128, 128, 1))
    pre = np.tile(pre, (1, 3))
    im = Image.fromarray(pre.astype(np.uint8))
    im.save(OUTPUT_FILE + "NonAugmented" + ".jpeg")

    i = 0
    for images, masks in ml.train_batches.take(20):
        image = np.array(images[0, :, :, 0])
        image = image.reshape((128, 128, 1))
        image = np.tile(image, (1, 3)).astype(np.uint8)

        im = Image.fromarray(image)
        im.save(OUTPUT_FILE + "augmented" + str(i) + ".jpeg")
        i += 1

