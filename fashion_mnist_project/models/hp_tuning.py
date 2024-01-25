import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import keras_tuner as kt


class SequentialModel(models.Sequential):
    def __init__(self, name=None):
        super().__init__(name=name)

    def score(self, x, y, sample_weight=None):

        return self.evaluate(x, y, verbose=0)[1]

    def fit(self, *args, **kwargs):
        return super().fit(*args, epochs=50, validation_split=0.1,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=5)], **kwargs)


class HyperModel(kt.HyperModel):

    def __init__(self, convolutional_layer_tuning=False, dense_layer_tuning=False, learning_rate_tuning=False,
                 convolutional_layer_hyperparameters=None, dense_layer_hyperparameters=None,
                 learning_rate_hyperparameters=None, cross_validation=False):
        self.convolutional_layer_tuning = convolutional_layer_tuning
        self.dense_layer_tuning = dense_layer_tuning
        self.learning_rate_tuning = learning_rate_tuning
        self.convolutional_layer_hyperparameters = convolutional_layer_hyperparameters
        self.dense_layer_hyperparameters = dense_layer_hyperparameters
        self.learning_rate_hyperparameters = learning_rate_hyperparameters
        self.cross_validation = cross_validation

    def build(self, hp):
        model = SequentialModel()

        if self.convolutional_layer_tuning:
            n_convolutional_layers = hp.Int('num_convpool_layers', min_value=1, max_value=3, step=1)
            n_filters = []
            filter_sizes = []
            for layer_num in range(n_convolutional_layers):
                n_filters.append(hp.Int(f'num_filters_{layer_num}', min_value=16 * (2 ** layer_num),
                                        max_value=64 * (2 ** layer_num), step=16 * (2 ** layer_num)))
                filter_sizes.append(hp.Int(f'filter_size_{layer_num}', min_value=3, max_value=5, step=2))

        elif self.cross_validation and not self.convolutional_layer_tuning:
            convolutional_architecture = hp.Choice('Convolutional Layers', values=[0, 1, 2]) # Note, for some reason when running the GridSearch Cross Validation, all combinations using the value 2 will not be generated. By editing the list to only include 2, the combinations will be generated.
            convolutional_architecture = self.convolutional_layer_hyperparameters[convolutional_architecture]
            n_convolutional_layers = convolutional_architecture.values['num_convpool_layers']
            n_filters = []
            filter_sizes = []
            for layer_num in range(n_convolutional_layers):
                n_filters.append(convolutional_architecture.values[f'num_filters_{layer_num}'])
                filter_sizes.append(convolutional_architecture.values[f'filter_size_{layer_num}'])

        else:
            n_convolutional_layers = 2
            n_filters = [64, 128]
            filter_sizes = [5, 3]

        if self.dense_layer_tuning:
            n_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
            n_units = []
            for layer_num in range(n_dense_layers):
                n_units.append(hp.Int(f'num_units_{layer_num}', min_value=128 / (2 ** layer_num),
                                      max_value=512 / (2 ** layer_num), step=128 / (2 ** layer_num)))

        elif self.cross_validation and not self.dense_layer_tuning:
            dense_architecture = hp.Choice('Dense Layers', values=[0, 1, 2])
            dense_architecture = self.dense_layer_hyperparameters[dense_architecture]
            n_dense_layers = dense_architecture.values['num_dense_layers']
            n_units = []
            for layer_num in range(n_dense_layers):
                n_units.append(dense_architecture.values[f'num_units_{layer_num}'])

        else:
            n_dense_layers = 1
            n_units = [256]

        if self.learning_rate_tuning:
            learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        elif self.cross_validation and not self.learning_rate_tuning:
            learning_rate = hp.Choice('Learning Rate', values=self.learning_rate_hyperparameters)

        else:
            learning_rate = 1e-4

        for layer_num in range(n_convolutional_layers):
            if layer_num == 0:
                model.add(
                    layers.Conv2D(n_filters[layer_num],
                                  kernel_size=filter_sizes[layer_num],
                                  activation='relu', input_shape=(28, 28, 1)))
            else:
                model.add(
                    layers.Conv2D(n_filters[layer_num],
                                  kernel_size=filter_sizes[layer_num],
                                  activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        for layer_num in range(n_dense_layers):
            model.add(
                layers.Dense(units=n_units[layer_num],
                             activation='relu'))

        model.add(layers.Dense(10))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model


    def score(self, hp, model, *args, **kwargs):
        return model.evaluate(*args, **kwargs)[1]


def tune_model(convolutional_layer_tuning=False, dense_layer_tuning=False, learning_rate_tuning=False):
    (x_train, y_train), _ = fashion_mnist.load_data()

    x_train = x_train / np.max(x_train)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    if convolutional_layer_tuning:
        directory = 'hyperparameter_tuning_output_convolution_layers'
        project_name = 'fashion_mnist_convolution_layers'
    elif dense_layer_tuning:
        directory = 'hyperparameter_tuning_output_dense_layers'
        project_name = 'fashion_mnist_dense_layers'
    elif learning_rate_tuning:
        directory = 'hyperparameter_tuning_output_learning_rate'
        project_name = 'fashion_mnist_learning_rate'
    else:
        directory = 'hyperparameter_tuning_output'
        project_name = 'fashion_mnist'

    tuner = kt.GridSearch(HyperModel(convolutional_layer_tuning, dense_layer_tuning, learning_rate_tuning),
                          objective='val_accuracy',
                          directory=directory,
                          project_name=project_name, overwrite=False)

    tuner.search(x_train, y_train, validation_data=(x_test, y_test))

    print(tuner.results_summary(num_trials=3))

