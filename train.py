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
from loadr import load_emnist_balanced_data
import mymodel

print(tf.__version__)


# fashion_mnist = keras.datasets.fashion_mnist
# print('data set fetched')
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels), class_names = load_emnist_balanced_data()

# train_images = train_images[:50000]
# train_labels = train_labels[:50000]
print("class names({}): {}".format(len(class_names), class_names))
print("train shape: {}, {}".format(train_images.shape, train_labels.shape))
print("test  shape: {}, {}".format(test_images.shape, test_labels.shape))
train_images = train_images / 255.0
test_images = test_images / 255.0

l = tf.keras.layers
data_format = 'channels_last'

input_shape = [28, 28, 1]

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = mymodel.create_model(input_shape, len(class_names))
model.summary()

model.fit(train_images, train_labels, batch_size=1000, epochs=2, callbacks = [cp_callback])
print('model fitted')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('model evaluated')
print('Test accuracy:', test_acc)
