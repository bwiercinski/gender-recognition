from __future__ import absolute_import, division, print_function

import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from create_test_data import get_test_data
from process_image import get_image, process_image

if __name__ == "__main__":
    print("Load data")
    with open("data/test_data/t_1000_200.pkl", "rb") as file:
        ((train_labels, train_data), (test_labels, test_data)) = pickle.load(file)
    # ((train_labels, train_data), (test_labels, test_data)) = get_test_data(1000, 200)

    print("Create model")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=train_data[0].shape),
        keras.layers.Dense(128, activation=tf.nn.relu6),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    print("Compile model")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("Fit model")
    model.fit(train_data, train_labels, epochs=5)

    print("Evaluate model")
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print("Test accuracy:", test_acc)

    print("My tests")
    my_image = [
        process_image(get_image("my_image/" + "Bartek_0001.jpg"), crop=False),
        process_image(get_image("my_image/" + "Kasia_0001.jpg"), crop=False),
        process_image(get_image("my_image/" + "Kasia_0002.jpg"), crop=False)
    ]
    predictions = model.predict(np.array(my_image))

    print("Kasia prediction:", np.argmax(predictions, axis=1))
