import numpy
import os
from keras.layers import Dense
from keras import Sequential

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("feature size: ", inputList.shape)
print("result size: ", resultList.shape)

model = Sequential()
model.add(Dense(50, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(inputList, resultList,
          epochs=200,
          batch_size=5,
          validation_split=0.2)
scores = model.evaluate(inputList, resultList)
print("metrics name: ", model.metrics_names)
print("\n")
print("\n %s: %.3f%%" % (model.metrics_names[1],
                         scores[1] * 100))
print("\n %s: %.3f" % (model.metrics_names[0],
                       scores[0]))
