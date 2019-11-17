import pandas as pd
import keras
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks, Sequential
import tensorflow as tf

csv = pd.read_csv('data/bmi.csv')
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

print(type(csv))
print(csv.head(n=20))

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

test_csv = csv[25000:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[25000:]

train_csv = csv[:25000]
train_pat = train_csv[['weight', 'height']]
train_ans = transformedLabel[:25000]

model = Sequential()
model.add(Dense(10, activation=tf.nn.relu, input_shape=(2,)))
model.add(Dense(3, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

tensorboard = callbacks.TensorBoard(log_dir='log', histogram_freq=0)
history = model.fit(train_pat, train_ans,
                    batch_size=50, epochs=50, verbose=1,
                    validation_data=(test_pat, test_ans),
                    callbacks=[tensorboard])
score = model.evaluate(test_pat, test_ans, verbose=0)
print(f"test loss={score[0]}, accuracy={score[1]}")
