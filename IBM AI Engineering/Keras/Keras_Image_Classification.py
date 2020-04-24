__author__ = 'compiler'

# Introduction
# Lets use the Keras library to build models for classificaiton problems.
# We will use the popular MNIST dataset, a dataset of images, for a change.
#
# The MNIST database, short for Modified National Institute of Standards and Technology database, is a large database of handwritten digits that is commonly used for training various image processing systems.
# The database is also widely used for training and testing in the field of machine learning.
#
# The MNIST database contains 60,000 training images and 10,000 testing images of digits written by high school students and employees of the United States Census Bureau.

import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

plt.imshow(X_train[0])


# With conventional neural networks, we cannot feed in the image as input as is.
# So we need to flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784.

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# Finally, before we start building our model, remember that for classification we need to divide our target variable into categories.
# We use the to_categorical function from the Keras Utilities package.

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

#Building Neural Network
# define classification model


def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and Test the Network:

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

model.save('classification_model.h5')
# Since our model contains multidimensional arrays of data, then models are usually saved as .h5 files


# When you are ready to use your model again, you use the load_model function from keras.models. See an example below:
# from keras.models import load_model
# pretrained_model = load_model('classification_model.h5')

# ----------------------------------------------------------------------------------------------------------------------
