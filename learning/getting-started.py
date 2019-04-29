from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow==2.0.0-alpha0

import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist

print("Load Dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Convert to Grayscale")
x_train, x_test = x_train / 255.0, x_test / 255.0



print("Build Sequential Model")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])


print("Compile Sequential Model")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Fit Training Data to Model")
model.fit(x_train, y_train, epochs=5)

print("Evaluation of Model")
model.evaluate(x_test, y_test)