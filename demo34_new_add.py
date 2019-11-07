import tensorflow as tf
import numpy as np

a1 = np.array([5, 3, 8])
a2 = np.array([3, -1, 2])
a3 = np.add(a1, a2)
print(a3)

b1 = tf.constant([5,3,8])
b2 = tf.constant([3,-1,2])
b3 = tf.add(b1, b2)
print(b3)

# sess = tf.Session()
# print("sess.run(b3): ",sess.run(b3))