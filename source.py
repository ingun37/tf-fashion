from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.pyplot as plt
from loadr import load_emnist_balanced_data
print(tf.__version__)


# fashion_mnist = keras.datasets.fashion_mnist
# print('data set fetched')
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels), class_names = load_emnist_balanced_data()

print("class names: {}".format(class_names))

train_images = train_images / 255.0
test_images = test_images / 255.0

# Here are layers of my deep learning architecture:

# 2D Convolutions (k=13, s=1, padding=valid)
# Max Pooling (k=2, s=2, padding=same)
# 2D Convolution (k=2, s=2, padding=same)
# Max Pooling (k=2, s=2, padding=same)
# 1x1 Convolutions (k=1, s=1, padding=valid)
# Fully Connected
# Fully Connected
model = keras.Sequential([
    keras.layers.Conv2D(16, (13, 13), activation=tf.nn.relu, input_shape=(28, 28, 1)), # 16x16x16
    keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'), # 8x8x16
    keras.layers.Conv2D(32, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu), #4x4x32
    keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'), #2x2x32
    keras.layers.Conv2D(64, (2,2), padding='same', activation=tf.nn.relu), #1x1x64
    keras.layers.Flatten(input_shape=(64, 1, 1)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])
print('model created')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('model compiled')
model.fit(train_images, train_labels, epochs=5)
print('model fitted')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('model evaluated')
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions)