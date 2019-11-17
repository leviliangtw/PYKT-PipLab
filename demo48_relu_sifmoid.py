import tensorflow as tf

v1 = [3.0, -1.0, 2.4, 5.9, 0.0001, 8.5, -0.0000000001]
r1 = tf.nn.relu(v1)
r2 = tf.nn.sigmoid(v1)
print(r1)
print(r2)