from tensorflow.keras.datasets import fashion_mnist
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def train_model():
    # Load the Fashion-MNIST dataset from TensorFlow
    (X_train, y_train), _ = fashion_mnist.load_data()

    # Reshape the data from 28x28 matrices (images) to 784-dimensional vectors
    X_train = X_train.reshape(60000, 784)

    # Normalize the data to have values between 0 and 1
    X_train = X_train / np.max(X_train)

    # Create a Logistic Regression model
    logisticRegression = LogisticRegression(random_state=37, max_iter=1000)

    # Create a dictionary of parameters to search for the best model using cross validation
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'multi_class': ['ovr', 'multinomial']}

    # Find the best model using 5-fold cross validation with GridSearchCV
    clf = GridSearchCV(logisticRegression, parameters, cv=5, verbose=10, n_jobs=-1, refit=True)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_

    with open('../../models/baseline.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Print the best parameters found
    print("The best parameters found for logistic regression:")
    print(clf.best_params_)
    print("The accuracy score obtained with the best parameters: " + str(clf.best_score_))


def load_model():
    with open('models/baseline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def predict_class(model, image):
    prediction = model.predict(image)
    predictionToClass = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return predictionToClass[prediction[0]]
