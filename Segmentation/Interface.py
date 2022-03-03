import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from Segmentation.Augmentations import Augmenter, AUGMENTATION_TYPE_DEMO, AUGMENTATION_SEED
from Segmentation.Model import unet_model
from Constants import BATCH_SIZE, BUFFER_SIZE, OUTPUT_CLASSES, EPOCHS, VAL_SUBSPLITS, MODEL_FILE


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class ML(tf.keras.layers.Layer):
    def __init__(self):
        # setup the model itself
        self.model = unet_model(output_channels=OUTPUT_CLASSES)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model_history = None
        self.train_num_examples = None
        self.validation_num_examples = None
        self.train_batches = None
        self.validation_batches = None
        self.sample_image = None
        self.sample_mask = None
        self.data_pipeline_training_configured = False
        self.data_pipeline_validation_configured = False

    def configure_training_data_pipeline(self, train_images, augmentations_type=AUGMENTATION_TYPE_DEMO, augmentations_seed=AUGMENTATION_SEED):
        self.data_pipeline_training_configured = True
        self.train_num_examples = len(train_images)
        augmenter = Augmenter(augmentations_type, augmentations_seed)
        self.train_batches = (
            train_images
                .cache()
                .shuffle(BUFFER_SIZE)
                .repeat()
                .map(lambda x, y: (augmenter.call(x, y)), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        for images, masks in self.train_batches.take(2):
            self.sample_image, self.sample_mask = images[0], masks[0]

    def configure_validation_data_pipeline(self, validation_images):
        self.validation_num_examples = len(validation_images)
        self.validation_batches = validation_images.batch(BATCH_SIZE)
        self.data_pipeline_validation_configured = True

    def verbose(self):
        # tf.keras.utils.plot_modParallelMapDatasetel(self.model, show_shapes=True)
        self.model.summary()

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
        if not(self.data_pipeline_validation_configured and self.data_pipeline_training_configured):
            print("Error; data pipeline is not configured for training")
        else:
            class DisplayCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(inner_self, epoch, logs=None):
                    clear_output(wait=True)
                    self.display_example(predictions=True, train_data=True, num=1)
                    print('\nSample Prediction after epoch {}\n'.format(epoch+1))

            steps_per_epoch = self.train_num_examples // BATCH_SIZE
            validation_steps = self.validation_num_examples // BATCH_SIZE//VAL_SUBSPLITS

            self.model_history = self.model.fit(self.train_batches, epochs=EPOCHS,
                                                steps_per_epoch=steps_per_epoch,
                                                validation_steps=validation_steps,
                                                validation_data=self.validation_batches,
                                                callbacks=[DisplayCallback()])

    def evaluate_model(self):
        if not self.data_pipeline_validation_configured:
            print("Error; data pipeline is not configured for evaluation")
        else:
            loss, acc = self.model.evaluate(self.validation_batches, batch_size=BATCH_SIZE, verbose=2)
            print("accuracy: {:5.2f}%".format(100 * acc))

    def make_prediction(self, image):
        return self.create_mask(self.model.predict(image[tf.newaxis, ...]))

    def save_model(self, filename, location=MODEL_FILE):
        self.model.save_weights(location + filename)

    def load_model(self, filename, location=MODEL_FILE):
        self.model.load_weights(location + filename)

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
