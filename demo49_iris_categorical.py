from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

df1 = read_csv("data/iris.data", header=None)
# print(type(df1), "\n", df1.head())
dataset = df1.values
# print(type(dataset), dataset[:3])
features = dataset[:, :4].astype(float)
labels = dataset[:, 4]
# print(type(features))
# print(type(labels))

encoder = LabelEncoder()
encoded_Y = encoder.fit(labels).transform(labels)
dummy_y = np_utils.to_categorical(encoded_Y)


# print(type(encoded_Y), encoded_Y[:10], encoded_Y[50:60], encoded_Y[100:110])
# print(type(dummy_y), dummy_y[:10], dummy_y[50: 60], dummy_y[100: 110])

def base_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# model = base_model()
estimator = KerasClassifier(build_fn=base_model, epochs=200,
                            batch_size=20, verbose=0)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=kfold)
print(f"accuracy:{results.mean()}, std:{results.std()}")
