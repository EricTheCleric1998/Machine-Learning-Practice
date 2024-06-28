import tensorflow as tf
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


model = tf.keras.Sequential([tf.keras.Input(shape=(28,28)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=512, activation='relu'),
                             tf.keras.layers.Dense(units=256, activation='relu'),
                             tf.keras.layers.Dense(units=128, activation='relu'),
                             tf.keras.layers.Dense(units = 10, activation='softmax')
                             ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=50, validation_split = 0.1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Model Accuracy")
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['train', 'val'], loc='upper left')
model.evaluate(x_test, y_test)

#CNN setup
model2 = tf.keras.Sequential([tf.keras.Input(shape = (28,28,1)),
                              tf.keras.layers.Conv2D(56,7),
                              tf.keras.layers.Activation('relu'),
                              tf.keras.layers.AveragePooling2D(7),
                              tf.keras.layers.Flatten(),
                              tf.keras.layers.Dense(units=512, activation='relu'),
                              tf.keras.layers.Dense(units=256, activation='relu'),
                              tf.keras.layers.Dense(units=128, activation='relu'),
                              tf.keras.layers.Dense(units=10, activation='softmax')
                              ])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()
history2 = model2.fit(x_train, y_train, epochs=100, validation_split = 0.1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
model2.evaluate(x_test, y_test)
plt.show()
