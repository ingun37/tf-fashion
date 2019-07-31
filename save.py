from __future__ import absolute_import, division, print_function, unicode_literals

import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
from loadr import digits_class_names
import mymodel
from PIL import Image
import matplotlib.pyplot as plt


print(tf.__version__)

class_names = digits_class_names()

data_format = 'channels_last'

input_shape = [28, 28, 1]

checkpoint_path = "training_digits/cp.ckpt"

model = mymodel.create_model(input_shape, len(class_names))
model.summary()

model.load_weights(checkpoint_path)

model.save('my_model.h5')