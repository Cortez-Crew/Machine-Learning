"""

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


print(f'Tensorflow Version: {tf.__version__}')

# Download the dataset
dataset_path = keras.utils.get_file('auto-mpg.data',
                                    'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
# print(f'Dataset Path: {dataset_path}')

# Specify the data columns
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

# import the dataset into a usable format
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
# print(dataset.tail())

# Drop any unknown values in the dataset
# print(dataset.isna().sum())
dataset.dropna()

# Set the values for the origin to have their own individual keys
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
# print(dataset.tail())

# Split the data into a train and evaluation set
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Look at the joint distribution of pairs of data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()

# Get a summary of the current data statistics
train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
print(train_stats)

# Separate the data from the 'labels' or the target values to be predicted
train_labels = train_dataset.pop('MPG')
test_dataset = test_dataset.pop('MPG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# Normalize the two sets of data for later
norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)


# Use a function to build the model since there will be a second one needed later
def build_model():
    bmodel = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    bmodel.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    return bmodel


model = build_model()
print(model.summary())

# Test the model with 10 examples from the training data
example_batch = norm_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


# Instead of using the built in progress bar, show the progress bar by dots
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(norm_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(f'\n{hist.tail()}')
