import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)

with tf.compat.v1.Session() as session1:
    # result = session1.run(c, feed_dict={
    #     a: [3, 4, 5],
    #     b: [-1, 2, 3]
    # })
    result = session1.run(c, {a: [1, 3], b: [2, 4]})
    print(result)
