import numpy as np
from keras import layers, models, Sequential
from keras.datasets import imdb

(train_data, train_label), (test_data, test_label) = \
    imdb.load_data(num_words=10000)
# print(train_data[0])
# print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
# print(type(word_index))
# print(word_index.keys())
print("word_index.items(): \n", word_index.items())

# for k in word_index.keys():
#     print(k)

reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
print("reverse_word_index: \n", reverse_word_index)


# for i in range(5):
#     decoded_first = ' '.join([reverse_word_index.get(i - 3, "?") for i in train_data[i]])
#     print(decoded_first)


def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
# y_train = vectorize_sequence(train_label).astype('float32') # not working
y_train = np.asarray(train_label).astype('float32')
y_test = vectorize_sequence(test_label).astype('float32')

model = Sequential()
model.add(layers.Dense(24, input_shape=(10000,), activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

x_val = x_train[:2500]
partial_x_train = x_train[2500:]
y_val = y_train[:2500]
partial_y_train = y_train[2500:]
history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=200, validation_data=(x_val, y_val))
