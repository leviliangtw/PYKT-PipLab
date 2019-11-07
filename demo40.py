import tensorflow as tf

tf.enable_eager_execution()

def compute_area(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


with tf.compat.v1.Session() as session1:
    area = compute_area(tf.constant([[5.0, 5.0, 5.0],
                                     [3.0, 4.0, 5.0],
                                     [2.3, 4.1, 4.8]]))
    result = session1.run(area)
    print(result)
