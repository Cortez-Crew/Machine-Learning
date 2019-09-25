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
print(f'Dataset Path: {dataset_path}')

# Specify the data columns
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']









