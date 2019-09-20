"""
This is my Basic Template for any of my Machine Learning Scripts
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

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
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

# The data is now equal lengths of 256
print('Review (train_data[0]):\n', train_data[0])
print('Lengths (train_data[1]):', len(train_data[1]), '(train_data[0]):', len(train_data[0]))

# The total possible word count
vocab_size = 10000

# TODO
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

