import numpy
import os
from keras.layers import Dense
from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("feature size: ", inputList.shape)
print("result size: ", resultList.shape)


def create_default_model(optimizer='adam', init='uniform'):
    model = Sequential()
    # 14, 8
    model.add(Dense(20, input_dim=8, kernel_initializer=init,
                    activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers = ['rmsprop', 'adam', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
param_grid = dict(optimizers=optimizers, epochs=epochs,
                  batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
