"""

"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version: " + tf.__version__)

# downloads and initializes the "Fashion MNIST database
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Divides the Integer values of train_images and test_images into grayscale values
train_images = train_images / 255
test_images = test_images / 255

# Creates a MatPlotLib view of the first 25 images in train_images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# # Display a single image from the data
# plt.figure()
# plt.imshow(train_images[0]) # Replace 0 with the desired index
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Reformats the images (2D arrays of pixels) into a single list (1D array of pixels)
# It is not organized as if you were to unstack layers of pixels
#
# The model also consists of teo Dense Layers
#   ~ The first layer is 128 Neurons that are using the 'Relu Function'
#   ~ The second layer is 10 Neurons that use the 'Softmax Function' which returns a probability between 0 and 1
#     on how likely the images is to be a certain thing
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

# The compile function takes the model and adds a few things to it
#   ~ Optimizer: A function telling the model how to update based on the data it sees and the loss function
#   ~ Loss Function: Measures how accurate the model is when training. Ideally this function wants to work
#     towards going to the minimum (closer to minimum = better accuracy)
#   ~ Metrics: A way to monitor the training model and tests. (The setting 'Accuracy' is the fraction of images
#     that are correctly classified)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

