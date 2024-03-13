# # the code below was imported from https://victorzhou.com/blog/keras-cnn-tutorial/

import numpy as np
import mnist
from tensorflow import keras
import os
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# # An alternative way of loading in the images
# train_images = mnist.train_images()
# train_labels = mnist.train_labels()
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

# Normalize the images.
train_images = (train_images / 255) - .5
test_images = (test_images / 255) - .5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

print(train_images.shape) # (60000, 28, 28, 1)
print(test_images.shape)  # (10000, 28, 28, 1)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Dropout

num_filters = 8 # number of convolutional network filters
filter_size = 3 # size of the convoluational network filters
pool_size = 2 # size of poling, in this case we select the max

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    Conv2D(num_filters, filter_size),
    MaxPooling2D(pool_size=pool_size),
    Dropout(.1),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

if 0: # training the model


    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
    )

    model.save_weights(os.path.join(os.getcwd(), 'cnn_keras_example_.weights.h5'))



else: # exercizing the model

    model.load_weights('cnn_keras_example_.weights.h5')
    predictions = model.predict(test_images[:20])
    print(np.argmax(predictions, axis = 1))
    print(test_labels[:20])

@tf.function
def fun():
    print(tf.one_hot([1,2,3,4], 5).get_value())

fun()