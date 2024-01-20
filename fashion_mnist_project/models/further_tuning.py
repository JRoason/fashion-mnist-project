import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import hp_tuning

conv_tuner = kt.GridSearch(hp_tuning.HyperModel(), objective='val_accuracy',
                           directory='hyperparameter_tuning_output_convlayer_threelayers',
                           project_name='fashion_mnist', overwrite=False)

convlayer_results = conv_tuner.get_best_hyperparameters(num_trials=3)[0:4]

dense_tuner = kt.GridSearch(hp_tuning.HyperModel(), objective='val_accuracy',
                            directory='hyperparameter_tuning_output_dense_layers',
                            project_name='fashion_mnist', overwrite=False)

dense_results = dense_tuner.get_best_hyperparameters(num_trials=3)[0:4]

learning_rates = [1e-2, 1e-3, 1e-4]

stop_early = EarlyStopping(monitor='val_loss', patience=5)


def generate_model(convlayer_result, dense_result, learning_rate):
    cnn_model = models.Sequential()
    for layer_num in range(convlayer_result.values['num_convpool_layers']):
        if layer_num == 0:
            cnn_model.add(
                layers.Conv2D(convlayer_result.values[f'num_filters_{layer_num}'],
                              kernel_size=convlayer_result.values[f'filter_size_{layer_num}'],
                              activation='relu', input_shape=(28, 28, 1)))
        else:
            cnn_model.add(
                layers.Conv2D(convlayer_result.values[f'num_filters_{layer_num}'],
                              kernel_size=convlayer_result.values[f'filter_size_{layer_num}'],
                              activation='relu'))
        cnn_model.add(layers.MaxPooling2D((2, 2)))

    cnn_model.add(layers.Flatten())

    for layer_num in range(dense_result.values['num_dense_layers']):
        cnn_model.add(
            layers.Dense(units=dense_result.values[f'num_units_{layer_num}'],
                         activation='relu'))

    cnn_model.add(layers.Dense(10))

    cnn_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    return cnn_model


(X_train, y_train), _ = fashion_mnist.load_data()

X_train = X_train / np.max(X_train)

y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

all_models = []

accuracies = []

kfold = KFold(n_splits=5, shuffle=True)

model_index = 0

for convlayer_result in convlayer_results:
    for dense_result in dense_results:
        for learning_rate in learning_rates:
            all_models.append([convlayer_result, dense_result, learning_rate])
            kfold_accuracies = np.zeros(5)
            kfold_index = 0
            for train_index, test_index in kfold.split(X_train, y_train):
                print(train_index, test_index)
                current_model = generate_model(convlayer_result, dense_result, learning_rate)
                current_model.summary()
                history = current_model.fit(X_train[train_index], y_train[train_index], epochs=50, validation_split=0.1, callbacks=[stop_early])
                score = current_model.evaluate(X_train[test_index], y_train[test_index], verbose=0)
                print(score[1])
                kfold_accuracies[kfold_index] = score[1]
                kfold_index += 1
            accuracies.append(np.mean(kfold_accuracies))
            model_index += 1

best_model = all_models[np.argmax(accuracies)]
best_model = generate_model(best_model[0], best_model[1], best_model[2])

best_model.summary()

best_model.save('models/best_model.h5')
