import numpy
import os
from keras.layers import Dense
from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("feature size: ", inputList.shape)
print("result size: ", resultList.shape)


def create_default_model():
    model = Sequential()
    # 14, 8
    model.add(Dense(20, input_dim=8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_default_model,
                        epochs=200, batch_size=50, verbose=False)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, inputList, resultList,
                          cv=fiveFold)

print(f"result mean={results.mean()}, "
      f"std: {results.std()}")
