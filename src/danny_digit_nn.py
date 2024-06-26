from danny import npClosest
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


model = tf.keras.Sequential([tf.keras.Input(shape = (28,28)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=256, activation='relu'),
                             tf.keras.layers.Dense(units=128, activation='relu'),
                             tf.keras.layers.Dense(units = 10, activation='softmax')
                             ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100)
model.evaluate(x_test, y_test)