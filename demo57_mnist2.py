import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from keras import datasets, utils, layers, Sequential
from keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(f"train image shape={train_images.shape}, test image shape={test_images.shape}")
print(f"train label shape={train_labels.shape}, test label shape={test_labels.shape}")
# print(matplotlib.get_backend())

flatten = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)
# print(TRAINING_SIZE, TEST_SIZE)
trainImages = np.reshape(train_images, (TRAINING_SIZE, flatten))
testImages = np.reshape(test_images, (TEST_SIZE, flatten))
print(f"after flatten, train image shape={trainImages.shape}, test image shape={testImages.shape}")

# convert to float
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
# print(trainImages[0])
trainImages /= 255
testImages /= 255
# print(trainImages[0])

NUM_DIGITS = len(np.unique(train_labels))
print(f'NUM_DIGITS={NUM_DIGITS}')
trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
testILabels = utils.to_categorical(test_labels, NUM_DIGITS)
model = Sequential()
model.add(layers.Dense(256, activation=tf.nn.relu, input_shape=(flatten,)))
model.add(layers.Dense(64, activation=tf.nn.relu))
model.add(layers.Dense(10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

# tbCallback = TensorBoard(log_dir='log', histogram_freq=True,
#                          write_graph=True, write_images=True)
#
# history = model.fit(trainImages, trainLabels, epochs=20, callbacks=[tbCallback])
# predictLabels = model.predict(testImages)
# print(f"test marker as: {testILabels[:10]}")
# print(f"predict as: {predictLabels[:10]}")
#
# loss, accuracy = model.evaluate(testImages, testILabels)
# print("\n")
# print(f'test loss={loss}, test accuracy={accuracy}')