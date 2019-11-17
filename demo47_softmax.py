import numpy as np
import tensorflow as tf

scores1 = [3.0, 1.0, 2.0]
scores2 = [4.0, 1.0, 1.0, 1.0]


def manualSoftMax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(manualSoftMax(scores1))
print(manualSoftMax(scores2))

print(tf.nn.softmax(scores1))
print(tf.nn.softmax(scores2))
