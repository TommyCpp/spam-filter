import tensorflow as tf

"""
Code below use two layer as the discriminator net and the  generator net
"""

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    curr_W, curr_b, loss = sess.run([W, b, loss], {x: [100, 200, 300, 400], y: [0, -1, -2, -3]})
    print(curr_W, curr_b, loss)
