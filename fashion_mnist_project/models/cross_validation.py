import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import keras_tuner as kt
import hp_tuning


def load_hyperparameters(n=3):
    convolutional_hyperparameters = kt.GridSearch(hp_tuning.HyperModel(), objective='val_accuracy',
                                                  directory='hyperparameter_tuning_output_convolutional_layers',
                                                  project_name='fashion_mnist',
                                                  overwrite=False).get_best_hyperparameters(num_trials=n)[0:n+1]
    dense_hyperparameters = kt.GridSearch(hp_tuning.HyperModel(), objective='val_accuracy',
                                          directory='hyperparameter_tuning_output_dense_layers',
                                          project_name='fashion_mnist',
                                          overwrite=False).get_best_hyperparameters(num_trials=n)[0:n+1]

    return convolutional_hyperparameters, dense_hyperparameters


def cross_validation(k=5):
    (x_train, y_train), _ = fashion_mnist.load_data()

    x_train = x_train / np.max(x_train)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

    convolutional_layer_hyperparameters, dense_layer_hyperparameters = load_hyperparameters(n=3)

    CVTuner = kt.tuners.SklearnTuner(kt.oracles.GridSearchOracle(kt.Objective('score', direction='max'), max_trials=30),
                                     hypermodel=hp_tuning.HyperModel(
                                         convolutional_layer_tuning=False,
                                         dense_layer_tuning=False,
                                         learning_rate_tuning=False,
                                         convolutional_layer_hyperparameters=convolutional_layer_hyperparameters,
                                         dense_layer_hyperparameters=dense_layer_hyperparameters,
                                         learning_rate_hyperparameters=[1e-2, 1e-3, 1e-4],
                                         cross_validation=True),
                                     cv=KFold(n_splits=k, shuffle=True),
                                     directory='hyperparameter_tuning_output_cross_validation',
                                     project_name='fashion_mnist',
                                     overwrite=False)


    CVTuner.search(x_train, y_train)

    best_hps = CVTuner.get_best_hyperparameters(num_trials=1)[0]

    convolutional_layer_params = convolutional_layer_hyperparameters[best_hps.values['Convolutional Layers']]
    dense_layer_params = dense_layer_hyperparameters[best_hps.values['Dense Layers']]
    learning_rate = best_hps.values['Learning Rate']

    model = models.Sequential()

    for layer_num in range(convolutional_layer_params.values['num_convpool_layers']):
        if layer_num == 0:
            model.add(
                layers.Conv2D(convolutional_layer_params.values[f'num_filters_{layer_num}'],
                              kernel_size=convolutional_layer_params.values[f'filter_size_{layer_num}'],
                              activation='relu', input_shape=(28, 28, 1)))
        else:
            model.add(
                layers.Conv2D(convolutional_layer_params.values[f'num_filters_{layer_num}'],
                              kernel_size=convolutional_layer_params.values[f'filter_size_{layer_num}'],
                              activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    for layer_num in range(dense_layer_params.values['num_dense_layers']):
        model.add(
            layers.Dense(units=dense_layer_params.values[f'num_units_{layer_num}'],
                         activation='relu'))

    model.add(layers.Dense(10))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss=CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    model.save('models/final_model_for_real.keras')

cross_validation(5)