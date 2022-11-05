import tensorflow as tf
from Constants import *
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

input_shape = [*INPUT_SHAPE, 3]  # [128, 128, 3]

initializer = tf.random_normal_initializer(0., 0.02)


def unet_model(output_channels:int):  # the originally modified model for my testing
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

    inputs = tf.keras.layers.Input(shape=input_shape)

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

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


def unet_encoder_module1(filters, size):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


def unet_upsample1(filters, size): # up sampler with one internal convolution
    filters = min(filters, 512)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(min(filters*2, 512), size,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    return result


def unmodified_unet(output_channels):
    inputs = tf.keras.layers.Input(shape=input_shape)
    concat = tf.keras.layers.Concatenate()
    pool = tf.keras.layers.MaxPool2D(pool_size=2)

    x1 = unet_encoder_module1(32, 3)(inputs)  # 128 layer
    x2 = unet_encoder_module1(64, 3)(pool(x1))  # 64 layer
    x3 = unet_encoder_module1(128, 3)(pool(x2))  # 32 layer
    x4 = unet_encoder_module1(256, 3)(pool(x3))  # 16 layer
    x5 = unet_encoder_module1(512, 3)(pool(x4))  # 8 layer
    x6 = unet_encoder_module1(512, 3)(pool(x5))  # 4 layer

    y5 = tf.keras.layers.Conv2DTranspose(512, 3, strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)(x6)  # 4->8

    # num filters passed is the return dimensionality (i.e. num after the upsample), internally ahs double as many
    y4 = unet_upsample1(256, 3)(concat([y5, x5]))  # 8->16
    y3 = unet_upsample1(128, 3)(concat([y4, x4]))  # 16->32
    y2 = unet_upsample1(64, 3)(concat([y3, x3]))  # 32->64
    y1 = unet_upsample1(32, 3)(concat([y2, x2]))  # 64->128

    done = unet_encoder_module1(32, 3)(concat([y1, x1]))

    last = tf.keras.layers.Conv2D(
        filters=output_channels, kernel_size=3,
        padding='same')(done)

    return tf.keras.Model(inputs=inputs, outputs=last)


def encoder_modified_unet(output_channels):
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

    inputs = tf.keras.layers.Input(shape=input_shape)
    concat = tf.keras.layers.Concatenate()

    top_skip = tf.keras.layers.Conv2D(32, 3,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False)(inputs)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    up_stack = [
        unet_upsample1(512, 3),  # 4x4 -> 8x8
        unet_upsample1(256, 3),  # 8x8 -> 16x16
        unet_upsample1(128, 3),  # 16x16 -> 32x32
        unet_upsample1(64, 3),   # 32x32 -> 64x64
    ]

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    last_upsample = unet_upsample1(32, 3)  # 64x64 -> 128x128
    x = last_upsample(x)

    # This is the last layer of the model
    x = concat([x, top_skip])
    x = tf.keras.layers.Conv2D(32, 3,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(output_channels, 3,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def encoder_decoder_modified_unet(output_channels):
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

    inputs = tf.keras.layers.Input(shape=input_shape)
    concat = tf.keras.layers.Concatenate()

    top_skip = tf.keras.layers.Conv2D(32, 3,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False)(inputs)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    last_upsample = pix2pix.upsample(32, 3)  # 64x64 -> 128x128
    x = last_upsample(x)

    # This is the last layer of the model
    x = concat([x, top_skip])

    x = tf.keras.layers.Conv2D(output_channels, 3,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False)(x)
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