import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

X_dim = 784
Z_dim = 100
h_dim = 128
mb_size = 32
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

# Discriminator Net


with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

with tf.name_scope('discriminator'):
    D_W1 = tf.Variable(tf.zeros(shape=[784, 128]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W2 = tf.Variable(tf.zeros(shape=[128, 1]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
with tf.name_scope('input'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')  # 100-dimension noise

with tf.name_scope('generator'):
    G_W1 = tf.Variable(tf.zeros(shape=[100, 128]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

    G_W2 = tf.Variable(tf.zeros(shape=[128, 784]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    with tf.name_scope('generator'):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        # todo: maybe change the sigmoid

        return G_prob


def discriminator(x):
    with tf.name_scope('discriminator'):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        # todo: maybe use the softmax instead of sigmoid

        return D_prob, D_logit


with tf.name_scope('D_loss'):
    G_sample = generator(Z)
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
with tf.name_scope('G_loss'):
    G_loss = -tf.reduce_mean(tf.log(D_fake))

# Only update D(X)'s parameters, so var_list = theta_D
with tf.name_scope('D_train'):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
with tf.name_scope('G_train'):
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])


i = 0

for it in range(10000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print(sess.run(theta_D))
        print(sess.run(theta_G))
