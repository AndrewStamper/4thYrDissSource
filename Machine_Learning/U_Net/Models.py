import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, concatenate  # , Cropping2D


class _Unit(tf.keras.layers.Layer):
    def __init__(self, num_convolutions=2, depth=4, size=3):
        assert num_convolutions > 0
        super().__init__()
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


# Define model
# noinspection PyCallingNonCallable
class ClassificationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.top_layer = _Layer(auto=UnetConfig(tiers=3, num_filters=4))
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, inp):
        c_1 = self.top_layer(inp)
        end_0 = self.flatten(c_1)
        end_1 = self.d1(end_0)
        return self.d2(end_1)


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
class SegmentationModel(tf.keras.Model):
    def __init__(self, config=UnetConfig()):
        super().__init__()
        self.top_layer = _Layer(auto=config)

    def call(self, inp):
        return self.top_layer(inp)


