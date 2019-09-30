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
# print(dataset)

# Drop any unknown values in the dataset
print(dataset.isna().sum())
dataset = dataset.dropna()

# Set the values for the origin to have their own individual keys
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())

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
test_labels = test_dataset.pop('MPG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# Normalize the two sets of data for later
norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)


# Use a function to build the model since there will be a second one needed later
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


model1 = build_model()
print(model1.summary())

# Test the model with 10 examples from the training data
example_batch = norm_train_data[:10]
example_result = model1.predict(example_batch)
print(example_result)


# Instead of using the built in progress bar, show the progress bar by dots
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

history1 = model1.fit(norm_train_data, train_labels, epochs=EPOCHS,
                      validation_split=0.2, verbose=0, callbacks=[PrintDot()])

# # Display the results from the trial history
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(f'\n{hist.tail()}')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error (MPG)')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error ($MPG^2$)')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()

    plt.show()


plot_history(history1)

# Since the model doesn't show much improvement after a while, lets stop the training early when improvement stops
model2 = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history2 = model2.fit(norm_train_data, train_labels, epochs=EPOCHS,
                      validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history2)


loss, mae, mse = model2.evaluate(norm_test_data, test_labels)
print('Testing Set Mean Abs Error: {:5.2f} MPG'.format(mae))

test_predictions = model2.predict(norm_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values (MPG)')
plt.ylabel('Prediction (MPG)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()
plt.clf()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error (MPG)')
plt.ylabel('Count')
plt.show()
