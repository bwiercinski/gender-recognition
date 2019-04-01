from __future__ import absolute_import, division, print_function

import pickle

import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    print('Load data')
    with open('data/test_data/t_1000_200.pkl', 'rb') as file:
        ((train_labels, train_data), (test_labels, test_data)) = pickle.load(file)

    print('Create model')
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    print('Compile model')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('Fit model')
    model.fit(train_data, train_labels, epochs=5)

    print('Evaluate model')
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print('Test accuracy:', test_acc)
