import tensorflow as tf
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from Constants import AUGMENTATION_SEED, AUGMENTATION_TYPE_NONE, AUGMENTATION_TYPE_DEMO


class Augmenter(tf.keras.layers.Layer):
    def __init__(self, augmentations=AUGMENTATION_TYPE_DEMO, seed=AUGMENTATION_SEED):
        super().__init__()
        self.seed = seed
        ia.seed(self.seed)
        if augmentations == AUGMENTATION_TYPE_NONE:
            self.seq = lambda x, y: (x, y)
        else:  # if == AUGMENTATION_TYPE_DEMO
            if augmentations != AUGMENTATION_TYPE_DEMO:
                print("Invalid augmentation type chosen, DEMO has been selected")
            self.seq = iaa.Sequential([
                iaa.Dropout([0.05, 0.2]),                     # drop 5% or 20% of all pixels
                iaa.Sharpen((0.0, 1.0)),                      # sharpen the image
                iaa.Affine(rotate=(-45, 45)),                 # rotate by -45 to 45 degrees (affects segmaps)
                iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
            ], random_order=True)

    def py_func(self, image, labels):
        # prepare numpy input
        np_image_pre = image.numpy()
        np_image_0_1 = np_image_pre / np_image_pre.max()
        np_image_0_255 = 255 * np_image_0_1
        np_image = np_image_0_255.astype(dtype=np.uint8)
        seg_map = SegmentationMapsOnImage(labels.numpy().astype(dtype=np.uint8), shape=labels.numpy().astype(dtype=np.uint8).shape)

        # calculate augmentations
        images_aug, labels_aug = self.seq(image=np_image, segmentation_maps=seg_map)
        array_labels_aug = labels_aug.get_arr()
        return images_aug, array_labels_aug

    def call(self, inputs, labels):
        im_shape = inputs.shape
        labels_shape = labels.shape
        images_aug, labels_aug = tf.py_function(self.py_func, (inputs, labels), [tf.float64, tf.float64])
        images_aug.set_shape(im_shape)
        labels_aug.set_shape(labels_shape)
        return images_aug, labels_aug
