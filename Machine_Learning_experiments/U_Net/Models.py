import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, concatenate, Input, ReLU, BatchNormalization  # , Cropping2D


class _Unit(tf.keras.layers.Layer):
    def __init__(self, num_convolutions=2, depth=4, size=3):
        assert num_convolutions > 0
        super().__init__()
        self.size = size
        self.depth = depth
        self.num_convolutions = num_convolutions
        if self.num_convolutions > 1:
            self.deeper = _Unit(num_convolutions-1, depth, size)
        else:
            self.deeper = None
        self.conv = Conv2D(depth, size, activation='relu', padding="same")

    def call(self, inp):
        if self.deeper is not None:
            x = self.deeper(inp)
        else:
            x = inp
        return self.conv(x)


# noinspection PyCallingNonCallable
class _Layer(tf.keras.layers.Layer):
    def __init__(self, auto=None, defined=None, scale=2):
        super().__init__()
        self.bottom_tier = False
        if defined is not None:
            self.left = defined[0]
            self.down = MaxPool2D(pool_size=scale)
            self.lower = defined[1]
            self.up = Conv2DTranspose(defined[2].depth, 1, activation='relu', strides=scale)
            self.right = defined[2]
        else:
            if auto is None:
                auto = UnetConfig()

            self.scale = auto.scale
            self.left = _Unit(num_convolutions=auto.num_convolutions, depth=auto.num_filters, size=auto.convolution_size)
            if auto.tiers > 1:
                self.down = MaxPool2D(pool_size=auto.scale)
                self.lower = _Layer(auto=UnetConfig(tiers=auto.tiers - 1, filter_ratio=auto.filter_ratio, scale=auto.scale, num_filters=auto.num_filters * auto.filter_ratio))
                self.up = Conv2DTranspose(auto.num_filters, 1, activation='relu', strides=auto.scale)
                self.right = _Unit(num_convolutions=auto.num_convolutions, depth=auto.num_filters, size=auto.convolution_size)
            else:
                self.bottom_tier = True

    def call(self, inp):
        pass_through = self.left(inp)
        if self.bottom_tier:
            return pass_through
        else:
            down_sampled = self.down(pass_through)
            lower_value = self.lower(down_sampled)
            up_ed = self.up(lower_value)
            # if using padding="valid" need to crop image
            # cropped = Cropping2D(cropping=((2, 2), (2, 2)))(pass_through)
            combined = concatenate([pass_through, up_ed], axis=3)
            return self.right.call(combined)


class UnetConfig:
    def __init__(self, tiers=3, filter_ratio=2, scale=2, num_filters=4, num_convolutions=2, convolution_size=3):
        self.tiers = tiers
        self.filter_ratio = filter_ratio
        self.scale = scale
        self.num_filters = num_filters
        self.num_convolutions = num_convolutions
        self.convolution_size = convolution_size


# Define model
# noinspection PyCallingNonCallable
class ClassificationModel(tf.keras.Model):
    def __init__(self, config=UnetConfig()):
        super().__init__()
        self.top_layer = _Layer(auto=config)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, inp):
        c_1 = self.top_layer(inp)
        end_0 = self.flatten(c_1)
        end_1 = self.d1(end_0)
        return self.d2(end_1)


# Define model
# noinspection PyCallingNonCallable
class SegmentationModel(tf.keras.Model):
    def __init__(self, config=UnetConfig(), dim=1):
        super().__init__()
        self.top_layer = _Layer(auto=config)
        self.to_features = Conv2D(dim, self.top_layer.left.size, activation='relu', padding="same")
        self.flatten = Flatten()

    def call(self, inp):
        x = self.top_layer(inp)
        x = self.to_features(x)
        x = self.flatten(x)
        return x



def testunit(inputs, filters = 3):
    conv1 = Conv2D(filters=filters, kernel_size=4, activation='relu', padding="same")
    conv2 = Conv2D(filters=filters, kernel_size=4, activation='relu', padding="same")

    conv1d = conv1(inputs)
    batch_norm1 = BatchNormalization()(conv1d)
    act1 = ReLU()(batch_norm1)

    conv2d = conv2(act1)
    batch_norm2 = BatchNormalization()(conv2d)
    act2 = ReLU()(batch_norm2)

    return act2

def testlayer(input_shape, down, filters = 3):
    inputs = Input(shape=input_shape)
    pool = MaxPool2D(pool_size=2)
    upconv = Conv2DTranspose(filters=filters, kernel_size=4, strides=2, activation='relu', padding="same")

    dl = testunit(inputs, filters=filters)
    sb = pool(dl)
    db = down(sb)
    ub = upconv(db)
    sr = concatenate([dl, ub], axis=3)
    out = testunit(sr, filters=filters)

    return tf.keras.Model(inputs=inputs, outputs=out)

def testmodel(output_channels:int):
    input_shape = [128, 128, 3]
    inputs = Input(shape=input_shape)

    bottom = Conv2D(filters=128, kernel_size=4, activation='relu', padding="same")  # 4x4
    up1 = testlayer([8, 8, 32], bottom, filters=64)
    up2 = testlayer([16, 16, 16], up1, filters=32)
    up3 = testlayer([32, 32, 8], up2, filters=16)
    up4 = testlayer([64, 64, 4], up3, filters=8)
    top = testlayer(input_shape, up4, filters=4)

    # This is the last layer of the model
    last = Conv2D(
        filters=output_channels, kernel_size=3,
        padding='same')

    out = last(top(inputs))

    return tf.keras.Model(inputs=inputs, outputs=out)





def testmodelold2(output_channels:int, config=UnetConfig()):
    inputs = Input(shape=[128, 128, 3])

    top_layer = _Layer(auto=config)
    to_features = Conv2D(filters=output_channels, kernel_size=top_layer.left.size, activation='relu', padding="same")

    x = top_layer.call(inputs)
    out = to_features(x)

    return tf.keras.Model(inputs=inputs, outputs=out)


def testmodelold(output_channels:int):
    inputs = Input(shape=[128, 128, 3])

    conv = Conv2D(filters=3, kernel_size=3, strides=2, activation='relu', padding="same")
    print(conv(inputs).shape)

    # This is the last layer of the model
    last = Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    out = last(conv(inputs))
    print(out.shape)

    return tf.keras.Model(inputs=inputs, outputs=out)