import tensorflow as tf

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
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

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

        return G_prob


def discriminator(x):
    with tf.name_scope('discriminator'):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

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

tf.summary.FileWriter('log/', sess.graph)