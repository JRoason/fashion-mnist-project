import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from fashion_mnist_project.models.hp_tuning import HyperModel
import keras_tuner as kt

# model = load_model('models/best_model.h5')
#
# model.summary()
#
# stop_early = EarlyStopping(monitor='val_loss', patience=5)
#
# (X_train, y_train), _ = fashion_mnist.load_data()
#
# X_train = X_train / np.max(X_train)
#
# y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))
#
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
#
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.show()
#
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.show()
#
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#
# print(test_acc)
# print(test_loss)
#
# model.save('models/final_model.keras')

def train_model():
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    (X_train, y_train), _ = fashion_mnist.load_data()

    X_train = X_train / np.max(X_train)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    model = models.Sequential()

    conv_tuner = kt.GridSearch(HyperModel(), objective='val_accuracy',
                               directory='hyperparameter_tuning_output_convlayer_threelayers',
                               project_name='fashion_mnist', overwrite=False)

    convlayer_results = conv_tuner.get_best_hyperparameters(num_trials=1)[0]

    dense_tuner = kt.GridSearch(HyperModel(), objective='val_accuracy',
                                directory='hyperparameter_tuning_output_dense_layers',
                                project_name='fashion_mnist', overwrite=False)

    dense_results = dense_tuner.get_best_hyperparameters(num_trials=1)[0]

    for layer_num in range(convlayer_results.values['num_convpool_layers']):
        if layer_num == 0:
            model.add(
                layers.Conv2D(convlayer_results.values[f'num_filters_{layer_num}'],
                              kernel_size=convlayer_results.values[f'filter_size_{layer_num}'],
                              activation='relu', input_shape=(28, 28, 1)))
        else:
            model.add(
                layers.Conv2D(convlayer_results.values[f'num_filters_{layer_num}'],
                              kernel_size=convlayer_results.values[f'filter_size_{layer_num}'],
                              activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    for layer_num in range(dense_results.values['num_dense_layers']):
        model.add(
            layers.Dense(units=dense_results.values[f'num_units_{layer_num}'],
                         activation='relu'))

    model.add(layers.Dense(10))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])

    test_acc, test_loss = model.evaluate(X_test, y_test, verbose=2)

    print(test_acc)
    print(test_loss)

    model.save('models/final_model_2.keras')

def load_model():
    model = tf.keras.models.load_model('models/final_model_2.keras')
    model.summary()
    return model


def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[np.argmax(prediction)]
