"""
Basic Classification tutorial from the TensorFlow website
This script will classify 28x28 images into different categories
https://www.tensorflow.org/tutorials/keras/basic_classification
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow and Keras
import tensorflow as tf

from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version: " + tf.__version__)

# Downloads and initializes the 'Fashion' MNIST database
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

# The model consists of 3 layers:
#   ~ Flatten: Reformats the images (2D arrays of pixels) into a single list (1D array of pixels)
#     It is not organized as if you were to unstack layers of pixels
#   ~ Dense (1): The first dense layer is 128 Neurons that are using the 'Relu Function'
#   ~ Dense (2): The second dense layer is 10 Neurons that use the 'Softmax Function' which returns a probability
#     between 0 and 1 on how likely the images is to be a certain thing
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

# Feeds in the training data into the model
# The model then learns to associate labels with certain images
# The model then makes predictions from the test data. Then the results are verified against the actual answers
model.fit(train_images, train_labels, epochs=5)

# Evaluate the results from the test data to see accuracy on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', (test_acc * 100), '%')

# Have the model make the predictions
predictions = model.predict(test_images)

# # Prints data about the index 0 prediction
# prediction0 = predictions[0]
# predictions0_ClassIndex = np.argmax(predictions[0])
# print('\n\n---\nPrediction data from the last 10 Nodes for item 0 in the test list:\n', prediction0)
# print('The Predicted category number', predictions0_ClassIndex)
# # noinspection PyTypeChecker
# print('The correct classname and category number: ', class_names[predictions0_ClassIndex])


# Plots the specified image
def plot_img(index, predictions_array, correct_label, img):
    predictions_array, correct_label, img = predictions_array[index], correct_label[index], img[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == correct_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[int(predicted_label)],
                                         100 * np.max(predictions_array),
                                         class_names[correct_label]), color=color)


# Plots a specified image showing statistics about the accuracy data for the specified test
def plot_value_array(index, predictions_array, correct_label):
    predictions_array, correct_label = predictions_array[index], correct_label[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[correct_label].set_color('blue')


# # Plots the image index specified along with it's data graph
# def specify_index_of_prediction():
#     index = 0
#     while index != 10:
#         index = int(input('What index would you like to see?'))
#         plt.figure(figsize=(6, 3))
#         plt.subplot(1, 2, 1)
#         plot_img(index, predictions, test_labels, test_images)
#         plt.subplot(1, 2, 2)
#         plot_value_array(index, predictions, test_labels)
#         plt.show()
#
#
# specify_index_of_prediction()


# Creates a plot for the first 15 test images
# It also shows the graph data
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, (2 * i) + 1)
    plot_img(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, (2 * i) + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()


# Grabs the specified image from the test dataset
# Then adds it to a batch of images
def predict():
    index = int(input('What index would you like to test?'))
    img = test_images[index]
    print('Shape of the single img', img.shape)

    img = (np.expand_dims(img, 0))
    print('Shape of the batch of imgs', img.shape)

    prediction = model.predict(img)
    print('Prediction data', prediction)

    plot_value_array(0, prediction, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    result = np.argmax(prediction[0])
    print('Prediction:', class_names[int(result)])


predict()
