import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print(train_data.shape)
print(train_labels.shape)
# print(test_data.shape)
# print(test_labels.shape)

# print(train_data[0])
# print(train_labels[0])
# print(test_data[0])
# print(test_labels[0])
# print("Breakpoint!")

# Make the Z table: mean=0, std=1
mean = train_data.mean(axis=0)  # y-axis, column
std = train_data.std(axis=0)  # y-axis, column
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std


# print("Breakpoint!, mean: ",mean)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.summary()
    return model


model = build_model()
# print(train_data.shape[1], 64*14)
model.fit(train_data, train_labels, validation_split=0.1,
          epochs=100, batch_size=10, verbose=1)