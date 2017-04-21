
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_points = 1000
vectors = []

for i in range(0, num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 + 0.1 + 0.3 + np.random.normal(0.0, 0.3)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]




w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = w * x_data + b

cost = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(0, 56000):
    sess.run(train)

print(sess.run(w), sess.run(b), sess.run(cost))


plt.plot(x_data, y_data, 'ro')
plt.plot(x_data,sess.run(w) * x_data + sess.run(b))
plt.show()