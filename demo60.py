import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant(100, name='a')
b = tf.constant(150, name='b')
multiple_op = a * b

# use v1 session api to enable graph logging
with tf.compat.v1.Session() as session1:
    with tf.compat.v1.summary.FileWriter("log",
                                         graph=session1.graph) as writer:
        pass
    print(session1.run(multiple_op))
