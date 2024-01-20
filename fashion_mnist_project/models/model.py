import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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


def load_model():
    return tf.keras.models.load_model('models/final_model.keras')


def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[np.argmax(prediction)]
