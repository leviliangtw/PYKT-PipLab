import tensorflow as tf

writer = tf.summary.create_file_writer('log')

with writer.as_default():
    for step in range(200):
        a = tf.constant(1, name='a')
        b = tf.constant(15, name='b')
        tf.summary.scalar("mymetric", a * b * step, step=step)
        writer.flush()