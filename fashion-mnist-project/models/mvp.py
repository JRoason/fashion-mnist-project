from tensorflow.keras.datasets import fashion_mnist
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# Load the Fashion-MNIST dataset from TensorFlow
(X_train, y_train), _ = fashion_mnist.load_data()

# Reshape the data from 28x28 matrices (images) to 784-dimensional vectors
X_train = X_train.reshape(60000, 784)

# Normalize the data to have values between 0 and 1
X_train = X_train / np.max(X_train)

# Create a Logistic Regression model
logisticRegression = LogisticRegression(random_state = 37, max_iter = 1000)

# Create a dictionary of parameters to search for the best model using cross validation
parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'multi_class': ['ovr', 'multinomial']}

# Find the best model using 5-fold cross validation with GridSearchCV
clf = GridSearchCV(logisticRegression, parameters, cv = 5, verbose = 10, n_jobs = -1, refit = True)
clf.fit(X_train, y_train)
best_model = clf.best_estimator_

with open('../../models/baseline.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Print the best parameters found
print("The best parameters found for logistic regression:")
print(clf.best_params_)
print("The accuracy score obtained with the best parameters: " + str(clf.best_score_))

# Split the data into training and validation sets for a dummy classifier
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=37)

# Create a dummy classifier that picks a class at random with equal probabilities
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(X_train, y_train)

# Print the accuracy of the dummy classifier
print("Accuracy of dummy classifier picking a class at random: " + str(dummy_clf.score(X_val, y_val)))