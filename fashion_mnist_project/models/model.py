import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


def train_model():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / np.max(x_train)
    x_test = x_test / np.max(x_test)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test = OneHotEncoder(sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    model = load_model('models/final_model_for_real.keras')

    history = model.fit(x_train, y_train, validation_split=0.1, epochs=50, verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(test_loss)
    print(test_acc)

    predictions = model.predict(x_test)

    plt.figure()

    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    plt.show()

    model.save('models/final_model_trained.keras')


def load_classifier():
    model = tf.keras.models.load_model('models/final_model_trained.keras')
    model.summary()
    return model

def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[np.argmax(prediction)]
