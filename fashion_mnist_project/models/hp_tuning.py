import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

import pickle

(X_train, y_train), _ = fashion_mnist.load_data()

X_train = X_train / np.max(X_train)

y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

class HyperModel(kt.HyperModel):
    def build(self, hp):
        model = models.Sequential()
        # for layer_num in range(hp.Int('num_convpool_layers', min_value=1, max_value=3, step=1)):
        #     if layer_num == 0:  # first layer should have input shape specified
        #         model.add(
        #             layers.Conv2D(hp.Int(f'num_filters_{layer_num}', min_value=16 * (2 ** layer_num),
        #                                  max_value=64 * (2 ** layer_num), step=16 * (2 ** layer_num)),
        #                           kernel_size=hp.Int(f'filter_size_{layer_num}', min_value=3, max_value=5, step=2),
        #                           activation='relu', input_shape=(28, 28, 1)))
        #     else:
        #         model.add(
        #             layers.Conv2D(hp.Int(f'num_filters_{layer_num}', min_value=16 * (2 ** layer_num),
        #                                  max_value=64 * (2 ** layer_num), step=16 * (2 ** layer_num)),
        #                           kernel_size=hp.Int(f'filter_size_{layer_num}', min_value=3, max_value=5, step=2),
        #                           activation='relu'))
        #     model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))  # The following layers and
        model.add(layers.MaxPooling2D((2, 2)))  # filter sizes were determined
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # by the search above
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        # for layer_num in range(hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)):
        #     model.add(
        #         layers.Dense(units=hp.Int(f'num_units_{layer_num}', min_value=128 / (2 ** layer_num),
        #                                   max_value=512 / (2 ** layer_num), step=128 / (2 ** layer_num)),
        #                      activation='relu'))

        model.add(layers.Dense(256, activation='relu'))  # Determined by search above

        model.add(layers.Dense(10))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        return model


conv_tuner = kt.GridSearch(HyperModel(), objective='val_accuracy',
                           directory='hyperparameter_tuning_output_convlayer_threelayers',
                           project_name='fashion_mnist', overwrite=False)

# tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

convlayer_results = conv_tuner.get_best_hyperparameters(num_trials=5)[0:6]

dense_tuner = kt.GridSearch(HyperModel(), objective='val_accuracy',
                            directory='hyperparameter_tuning_output_dense_layers',
                            project_name='fashion_mnist', overwrite=False)

dense_results = dense_tuner.get_best_hyperparameters(num_trials=5)[0:6]

print(conv_tuner.results_summary(num_trials=5))
print(dense_tuner.results_summary(num_trials=5))

for i in range(5):
    print(f"""
    The {i}th set of hyperparameters is:
    {convlayer_results[i].values}
    """)

