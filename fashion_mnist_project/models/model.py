import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
import pickle


def train_model():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / np.max(x_train)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

    model = load_model('models/final_model_for_real.keras')

    history = model.fit(x_train, y_train, validation_split=0.1, epochs=50, verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

    model.save('models/final_model_trained.keras')


def load_classifier(cnn=True):
    if cnn:
        model = tf.keras.models.load_model('models/model.keras')
    else:
        with open('models/baseline.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[np.argmax(prediction)]
