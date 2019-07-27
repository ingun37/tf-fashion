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
from loadr import class_names
import mymodel
from PIL import Image
import matplotlib.pyplot as plt


print(tf.__version__)

class_names = class_names()

data_format = 'channels_last'

input_shape = [28, 28, 1]

checkpoint_path = "training_1/cp.ckpt"

model = mymodel.create_model(input_shape, len(class_names))
model.summary()

model.load_weights(checkpoint_path)

test_img_path = os.path.join(os.path.dirname(__file__), "test.jpeg")

test_img_raw = Image.open(test_img_path).resize((28, 28))
test_rgb = np.array(test_img_raw)
test_gray = 1 - np.dot(test_rgb[...,:3], [0.2989, 0.5870, 0.1140]).reshape((28, 28, 1)) / 255.0
print(test_gray)
plt.figure()
plt.imshow(test_gray.reshape((28, 28)))
plt.colorbar()
plt.grid(False)
plt.show()

tests_gray = np.array([test_gray])
print("test input shape: {}".format(tests_gray.shape))
predictions = model.predict(tests_gray)

result = sorted(zip(predictions[0], class_names))
print(list(result))
