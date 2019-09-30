"""

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

print(f'Tensorflow Version: {tf.version.VERSION}')

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Only use the first 1000 data objects for this
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model_ = create_model()
model_.summary()

# Create a file to save the training data
ckpt_path = 'saving_models/training_1/.ckpt'
ckpt_dir = os.path.dirname(ckpt_path)

# Create callback for the model
ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, verbose=1)

model_.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[ckpt_callback])








