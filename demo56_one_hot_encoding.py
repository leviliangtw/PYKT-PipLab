from keras import utils

orig1 = 5
NUM_DIGITS = 10
# Converts a class vector (integers) to binary class matrix.
encode1 = utils.to_categorical(orig1, NUM_DIGITS)
print(f"value = {orig1}, one hot encoded={encode1}")

orig2 = [1, 2, 7 ,10]
NUM_NEWS = 11
# Converts a class vector (integers) to binary class matrix.
encode2 = utils.to_categorical(orig2, NUM_NEWS)
print(f"value = {orig2}, one hot encoded={encode2}")
