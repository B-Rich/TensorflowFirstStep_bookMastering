import tensorflow as tf


#y = tf.sqrt(a)
#y = tf.pow(a, b)
#y = tf.exp(a)
#y = tf.log(a)
#y = tf.maximum(a, b)
#y = tf.minimum(a, b)
#y = tf.cos(90.0)
#y = tf.sin(30.0)

#y = tf.matmul(a, b)
#y = tf.diag(b)
#y = tf.transpose(b)
#y = tf.matrix_determinant(b)
#y = tf.matrix_inverse(b)

sess = tf.Session()

print(sess.run(y, feed_dict={a: [[1, 2, 3]], b: [[1, 2, 3], [2, 3, 4], [6, 4, 5]]}))