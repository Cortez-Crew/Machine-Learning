"""
This is my Basic Template for any of my Machine Learning Scripts
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Version:', tf.__version__)
print('Numpy version', np.__version__)

# Downloads and initializes the 'imdb' database
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Data is split up into reviews with a label of 1 for positive for a 0 for negative
# The reviews are lists of integers that correspond to words in the dictionary
# The reviews all vary in length which is going to be an issue for the network since inputs must be the same length
print('Training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))
print('Review  - (train_data[0]):', train_data[0])
print('Lengths (train_data[1]):', len(train_data[1]), '(train_data[0]):', len(train_data[0]))

# A dictionary mapping integers to their corresponding words
word_index = imdb.get_word_index()

# The first few indices are reserved
# The word_index must be shifted then
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Flip the orientation of the word index so that the reserved indices are at the end
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Replaces all the integer values in a review with their corresponding readable words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))

# Change the training data to be padded with <PAD> sequences to make all data equal size
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'],
                                                        padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'],
                                                       padding='post', maxlen=256)

# The data is now equal lengths of 256
print('Review (train_data[0]):\n', train_data[0])
print('Lengths (train_data[1]):', len(train_data[1]), '(train_data[0]):', len(train_data[0]))

# The total possible word count
vocab_size = 10000

# The model consists of 4 layers:
#   ~ Embedding: The input is linked up with its integer-encoding of the vocabulary and looks up the word-index
#     This makes the output array look like (batch, sequence, embedding)
#   ~ GlobalAveragePool1D: This makes it so the output vector is a fixed length averaging, This allows the model to deal
#     with variable inputs in a simple way
#   ~ Dense (1): The first dense layer is 16 hidden neurons that use the 'Relu Function'
#   ~ Dense (2): The second dense layer is a single result of 1 or 0, using the 'Sigmoid function' to determine the
#     probability and confidence of the choice
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Prints a pretty thing that shows info about the model
model.summary()

# The compile function takes the model and adds a few things to it
#   ~ Optimizer: A function telling the model how to update based on the data it sees and the loss function
#   ~ Loss Function: Measures how accurate the model is when training. Ideally this function wants to work
#     towards going to the minimum (closer to minimum = better accuracy)
#   ~ Metrics: A way to monitor the training model and tests. (The setting 'Accuracy' is the fraction of images
#     that are correctly classified)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Sets apart the first 10,000 reviews in the training data to use as a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model for 40 epochs in mini-batches of 512 samples.
# This is 40 iterations over all samples in the x_train and y_train tensors.
# While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set:

# Feeds in the training data into the model
# The model does this over 40 iterations of the training data
# While training it is also compares to the consistent validation data
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Evaluate the model and display the results
results = model.evaluate(test_data, test_labels)
print(results)

# Use the dictionary provided by the model history to grab the results from the training data
# dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Create two plots to show the results of the Training ans Test data
#
epochs = range(1, len(acc) + 1)
# 'bo' means blue dot
plt.plot(epochs, loss, 'bo', label='Training Loss')
# 'b' means solid blue line
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# clear the first plot and begin the second one
plt.clf()
plt.plot(epochs, acc, 'ro', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
