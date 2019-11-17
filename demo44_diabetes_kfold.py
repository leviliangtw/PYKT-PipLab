import numpy
import os
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import StratifiedKFold

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("feature size: ", inputList.shape)
print("result size: ", resultList.shape)

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalscores = []

global model
model = Sequential()
# 14, 8
model.add(Dense(50, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for train, test in fiveFold.split(inputList, resultList):
    model.fit(inputList[train], resultList[train],
              epochs=200, batch_size=50, verbose=False)

    scores = model.evaluate(inputList[test], resultList[test],
                            verbose=False)
    print("metrics name: ", model.metrics_names)
    print("%s: %.3f%%" % (model.metrics_names[1],
                             scores[1] * 100))
    print("%s: %.3f" % (model.metrics_names[0],
                           scores[0]))
    totalscores.append(scores[1] * 100)

print(f"score average: {numpy.mean(totalscores)}, "
      f"std: {numpy.std(totalscores)}")
