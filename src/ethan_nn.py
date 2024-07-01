import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

path = r'../data/PhiUSIIL_Phishing_URL_Dataset.csv'

# To help with understanding, print the absolute path
print(os.path.abspath(path))

data = np.genfromtxt(path, delimiter=",", dtype=str)
print(data.shape)

header_row = data[0, :]
print(f'header_row = {header_row}')

# remove header
data = data[1:, :]
first_row = data[0, :]
print(f'first_row={first_row}')

# the label is in column 49. Therefore, the next line will put all the labels into a vector.
label = data[:, 49].astype(np.float64)
print(f'first 15 labels = {label[0:15]}')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
#
# model.save('Test.keras')

model = tf.keras.models.load_model('Test.keras')

loss, accuracy = model.evaluate(x_test, y_test)

plt.imshow(x_train[0])
plt.show()

predictions = model.predict([x_test])

print(loss)
print(accuracy)
print(predictions)