import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

def train_model():
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    (X_train, y_train), _ = fashion_mnist.load_data()

    X_train = X_train / np.max(X_train)

    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    model = models.Sequential()

    model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(10))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])

    model.save('models/final_model_2.keras')

def load_model():
    model = load_model('models/final_model_2.keras')
    model.summary()
    return model


def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[np.argmax(prediction)]
