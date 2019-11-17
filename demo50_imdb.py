import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(type(X_train), type(y_train), type(X_test), type(y_test))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(np.unique(y_train))
print(np.unique(y_test))

# concat
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
print(X.shape, y.shape)
print(len(np.unique(np.hstack(X))))

