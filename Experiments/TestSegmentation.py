import tensorflow as tf
from Machine_Learning.U_Net.Unet import Unet, UnetConfig
from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x = x_train
y = y_train > 0.2

# Define a new instance of the UNet
config = UnetConfig(tiers=1, filter_ratio=2, scale=2, num_filters=2, num_convolutions=1, convolution_size=3)
u_net = Unet(s_dim=1, config=config)
u_net.run(x, x, x, x, batch=1, epochs=5)


a = x_train[0]
b = u_net.get_mask(a)

aim = UltrasoundScan(a)
bim = UltrasoundScan(b)

aim.write_image("test1")
bim.write_image("test2")


