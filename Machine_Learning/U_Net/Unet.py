import tensorflow as tf
from Machine_Learning.U_Net.Models import UnetConfig, ClassificationModel, SegmentationModel


class Unet:
    def __init__(self, mode="Segmentation", s_dim=1, config=UnetConfig()):

        if mode == "Segmentation":
            # Create an instance of the model
            self.model = SegmentationModel(dim=s_dim, config=config)
            # Define Loss
            self.loss_object = tf.keras.losses.CategoricalCrossentropy()
            # Define loss and accuracy
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
            self.validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')

        elif mode == "Classification":
            # Create an instance of the model
            self.model = ClassificationModel(config=config)
            # Define Loss
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # Define loss and accuracy
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
            self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

        else:
            raise ValueError("u_net model is not supported")

        # Pick optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Record number of epochs ran
        self.epoch = 0

    # Training function
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # noinspection PyCallingNonCallable
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Testing Function
    @tf.function
    def validation_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # noinspection PyCallingNonCallable
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.validation_loss(t_loss)
        self.validation_accuracy(labels, predictions)

    # function to run one epoch
    def one_epoch(self, train_ds, validate_ds):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.validation_loss.reset_states()
        self.validation_accuracy.reset_states()

        for images, labels in train_ds:
            self.train_step(images, labels)

        for validate_images, validate_labels in validate_ds:
            self.validation_step(validate_images, validate_labels)

        self.epoch = self.epoch + 1
        print(
            f'Epoch {self.epoch}, '
            f'Loss: {self.train_loss.result()}, '
            f'Accuracy: {self.train_accuracy.result() * 100}, '
            f'Validate Loss: {self.validation_loss.result()}, '
            f'Validate Accuracy: {self.validation_accuracy.result() * 100}'
        )
        print("weights:", len(self.model.weights))
        print("trainable weights:", len(self.model.trainable_weights))

    def run(self, x_train, y_train, x_test, y_test, batch=32, epochs=5):
        y_train = y_train.reshape((y_train.shape[0], -1))
        y_test = y_test.reshape((y_test.shape[0], -1))


        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add a channels dimension
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        # Shuffle the data and put into batches
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(batch)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)

        # Run for a number of Epochs
        for epoch in range(epochs):
            self.one_epoch(train_ds, test_ds)

    def get_mask(self, x):
        # noinspection PyCallingNonCallable
        y = self.model(x, training=False)
        return y.reshape((*x.shape, -1))



