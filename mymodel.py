import tensorflow as tf
from tensorflow import keras

def create_model(input_shape, class_num)-> keras.models.Sequential:
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(class_num, activation='softmax'))

    print('model created')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print('model compiled')
    return model