from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

from tensorflow.examples.tutorials.mnist import input_data
from termcolor import colored, cprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape=[None, 784])
x_ = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def conv_1(x, W):
    return tf.nn.conv2d(x, W, strides=[2,2], padding='VALID')


def conv_2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Define the first convolution layer here
W_conv1 = weight_variable([12,12,1,25])
b_conv1 = bias_variable([25])
h_conv1 = tf.nn.relu(conv_1(x_,W_conv1))+ b_conv1

# Define the second convolution layer here
W_conv2 = weight_variable([5,5,25,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv_2(h_conv1,W_conv2))+ b_conv2

# Define maxpooling
h_pool2 = max_pool_2x2(h_conv2)

# All subsequent layers will be fully connected ignoring geometry so we'll flatten the layer
# Flatten the h_pool2_layer (as it has a multidimensiona shape)
h_pool2_flat = tf.reshape(h_pool2, [-1,5*5*64])

# Define the first fully connected layer here
W_fc1 = weight_variable([5*5*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Use dropout for this layer (should you wish)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# The final fully connected layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# We'll use the cross entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# And classification accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# And the Adam optimiser
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# Load the mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Let us visualise the first 16 data points from the MNIST training data

fig = plt.figure()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(mnist.train.images[i].reshape(28, 28), cmap='Greys_r')


# Start a tf session and run the optimisation algorithm
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i%500 == 0:
        train_accuracy = accuracy.eval(session=sess ,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess , feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Print accuracy on the test set
print ('Test accuracy: %g' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# Visualise the filters in the first convolutional layer
with sess.as_default():
    W = W_conv1.eval()

# Add code to visualise filters here
fig = plt.figure()
for i in range(25):

    ax = fig.add_subplot(5, 5, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(W[:,:,0,i], cmap='Greys_r')

H =  sess.run(h_conv1, feed_dict={x: mnist.test.images})

# Add code to visualise patches in the test set that find the most result in
# the highest activations for filters 0, ... 4

for j in range(5):
    #pick some 5 from the 25&cut h
    H_cut=H[:,:,:,j]

    #calculate the args
    H_sort = np.argsort(H_cut,axis=None)

    print(str(j) + "th filter")

    fig = plt.figure()
    for i in range(12):
        index_flat=np.where(H_sort == i)
        index = np.unravel_index(index_flat,H_cut.shape)

        image = index[0][0][0]
        x_coord = index[1][0][0]*2
        y_coord = index[2][0][0]*2

        patch = mnist.test.images.reshape(10000, 28, 28)[image, x_coord:x_coord+12, y_coord:y_coord+12]

        ax = fig.add_subplot(3, 4, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(patch, cmap='Greys_r')




