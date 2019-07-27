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

# train_images = train_images[:50000]
# train_labels = train_labels[:50000]
print("class names({}): {}".format(len(class_names), class_names))
print("train shape: {}, {}".format(train_images.shape, train_labels.shape))
print("test  shape: {}, {}".format(test_images.shape, test_labels.shape))
train_images = train_images / 255.0
test_images = test_images / 255.0

l = tf.keras.layers
data_format = 'channels_last'
max_pool = l.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

input_shape = [28, 28, 1]

model = tf.keras.Sequential()
model.add(l.Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
model.add(l.BatchNormalization())
model.add(l.Conv2D(32, kernel_size = 3, activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Dropout(0.4))

model.add(l.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Dropout(0.4))

model.add(l.Conv2D(128, kernel_size = 4, activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Flatten())
model.add(l.Dropout(0.4))
model.add(l.Dense(len(class_names), activation='softmax'))

print('model created')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('model compiled')
model.fit(train_images, train_labels, batch_size=1000, epochs=2)
print('model fitted')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('model evaluated')
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions)