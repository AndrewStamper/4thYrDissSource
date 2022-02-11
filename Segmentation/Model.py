import tensorflow as tf
from Constants import *
from tensorflow_examples.models.pix2pix import pix2pix

input_shape = [*INPUT_SHAPE, 3]  # [128, 128, 3]

base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = True

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)





def testunit(inputs, filters = 3):
    conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, activation='relu', padding="same")
    conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, activation='relu', padding="same")

    conv1d = conv1(inputs)
    batch_norm1 = tf.keras.layers.BatchNormalization()(conv1d)
    act1 = tf.keras.layers.ReLU()(batch_norm1)

    conv2d = conv2(act1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(conv2d)
    act2 = tf.keras.layers.ReLU()(batch_norm2)

    return act2

def testlayer(input_shape, down, filters = 3):
    inputs = tf.keras.layers.Input(shape=input_shape)
    pool = tf.keras.layers.MaxPool2D(pool_size=2)
    upconv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, activation='relu', padding="same")

    dl = testunit(inputs, filters=filters)
    sb = pool(dl)
    db = down(sb)
    ub = upconv(db)
    sr = tf.keras.layers.concatenate([dl, ub], axis=3)
    out = testunit(sr, filters=filters)

    return tf.keras.Model(inputs=inputs, outputs=out)


def testmodel(output_channels:int):
    input_shape = [128, 128, 3]
    inputs = tf.keras.layers.Input(shape=input_shape)

    bottom = tf.keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu', padding="same")  # 4x4
    up1 = testlayer([8, 8, 32], bottom, filters=64)
    up2 = testlayer([16, 16, 16], up1, filters=32)
    up3 = testlayer([32, 32, 8], up2, filters=16)
    up4 = testlayer([64, 64, 4], up3, filters=8)
    top = testlayer(input_shape, up4, filters=4)

    # This is the last layer of the model
    last = tf.keras.layers.Conv2D(
        filters=output_channels, kernel_size=3,
        padding='same')

    out = last(top(inputs))

    return tf.keras.Model(inputs=inputs, outputs=out)