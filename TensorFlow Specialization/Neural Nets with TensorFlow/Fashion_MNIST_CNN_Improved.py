__author__= "compiler"

import tensorflow as tf
from os import path, getcwd, chdir
import numpy as np


data = np.load('mnist.npz')
print(data.files)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


def train_mnist_conv():

  class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
      if (logs.get('accuracy') > 0.998):
        print("\nReached 99.8% accuracy so cancelling training!")

        self.model.stop_training = True


  training_images = data['x_train']
  training_labels = data["y_train"]
  test_images = data["x_test"]
  test_labels = data["y_test"]

  training_images = training_images / 255.0
  test_images = test_images / 255.0
  training_images = training_images.reshape((-1, 28, 28, 1))

  callbacks = myCallback()

  model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                                             input_shape=(28, 28, 1)),
                                      tf.keras.layers.MaxPooling2D(2, 2),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(128, activation='relu'),
                                      tf.keras.layers.Dense(10, activation='softmax')
                                      ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # model fitting
  history = model.fit(training_images,
                      training_labels,
                      epochs=20,
                      callbacks=[callbacks])
  # model fitting
  return history.epoch, history.history['accuracy'][-1]


_, _ = train_mnist_conv()
