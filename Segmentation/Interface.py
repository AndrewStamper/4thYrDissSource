import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt

from Segmentation.surrounding import inputs, display
from Segmentation.Augmentations import Augment
from Segmentation.Model import unet_model
from Constants import BATCH_SIZE, BUFFER_SIZE, OUTPUT_CLASSES, EPOCHS, VAL_SUBSPLITS


class ML(tf.keras.layers.Layer):
    def __init__(self, train_images, validation_images, info):
        self.info = info

        # configure batches
        self.train_batches = (
            train_images
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .repeat()
                .map(Augment())
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        self.validation_batches = validation_images.batch(BATCH_SIZE)

        # setup the model itself
        self.model = unet_model(output_channels=OUTPUT_CLASSES)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        tf.keras.utils.plot_model(self.model, show_shapes=True)
        self.model_history = None

        # setup sample
        for images, masks in self.train_batches.take(2):
            self.sample_image, self.sample_mask = images[0], masks[0]

    @staticmethod
    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    def display_example(self, predictions=False, train_data=True, num=1):
        if predictions:
            if train_data:
                display([self.sample_image, self.sample_mask,
                         self.create_mask(self.model.predict(self.sample_image[tf.newaxis, ...]))])
            else:
                for image, mask in self.validation_batches.take(num):
                    pred_mask = self.model.predict(image)
                    display([image[0], mask[0], self.create_mask(pred_mask)])
        else:
            display([self.sample_image, self.sample_mask])

    def train(self):
        class DisplayCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(innerself, epoch, logs=None):
                clear_output(wait=True)
                self.display_example(predictions=True, train_data=True, num=1)
                print('\nSample Prediction after epoch {}\n'.format(epoch+1))

        steps_per_epoch = self.info.splits['train'].num_examples // BATCH_SIZE
        validation_steps = self.info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

        self.model_history = self.model.fit(self.train_batches, epochs=EPOCHS,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_steps=validation_steps,
                                            validation_data=self.validation_batches,
                                            callbacks=[DisplayCallback()])

    def show_epoch_progression(self):
        loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']

        plt.figure()
        plt.plot(self.model_history.epoch, loss, 'r', label='Training loss')
        plt.plot(self.model_history.epoch, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()


train_imageso, test_imageso, infoo = inputs()
ml = ML(train_imageso, test_imageso, infoo)
ml.display_example()
ml.train()
ml.show_epoch_progression()
ml.display_example(predictions=True, train_data=False, num=3)
