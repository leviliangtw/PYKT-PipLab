import tensorflow as tf


def add(p, q):
    return tf.math.add(p, q)


print(add([1, 2, 3], [4, 5, 6]))
print(add([1, 2.0, 3], [4.0, 5, 6]))
print(add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
# print(add([1,2,3], [4,5,6.0])) # line line will crash
