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

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()

plt.close()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
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