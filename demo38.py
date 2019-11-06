import tensorflow as tf


@tf.function  # performance is better than demo37
def add(p, q):
    return p + q


print(add([1, 2, 3, 4], [4, 5, 6, 7]))
