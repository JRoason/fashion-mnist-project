import numpy as np
import mnist_reader
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

kf = KFold(n_splits=5)

X_train = X_train / np.max(X_train)

clf = LogisticRegression(multi_class='multinomial', max_iter = 1000)

scores = cross_val_score(clf, X_train, y_train, cv=kf)

print("Cross validation scores: ", scores)
print("Mean cross validation score: ", np.mean(scores))
print("Standard deviation of cross validation scores: ", np.std(scores))
