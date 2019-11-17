import numpy
import os
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import train_test_split

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("feature size: ", inputList.shape)
print("result size: ", resultList.shape)
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.2, random_state=3579)

model = Sequential()
# 14, 8
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(feature_train, label_train,
          epochs=200,
          batch_size=50)
scores = model.evaluate(feature_test, label_test)
print("metrics name: ", model.metrics_names)
print("\n")
print("\n %s: %.3f%%" % (model.metrics_names[1],
                         scores[1] * 100))
print("\n %s: %.3f" % (model.metrics_names[0],
                       scores[0]))
