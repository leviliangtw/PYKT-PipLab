import tensorflow as tf
from tensorflow import compat

compat.v1.disable_eager_execution()
t1 = tf.constant('hello tensorflow')
session1 = compat.v1.Session()
print(session1.run(t1))