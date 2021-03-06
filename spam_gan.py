# coding=utf-8
# read data
from extra_email import *
import tensorflow as tf

SOURCE_MSG = "./data/source/msg.txt"
SOURCE_SPAM = "./data/source/spam.txt"
TEST_MSG = "./data/source/test/msg.txt"
TEST_SPAM = "./data/source/test/spam.txt"
embedding_size = 32
N_WORDS = 8


# batcher
class BatchManager:
    def __init__(self, emails: list):
        self.index = 0
        self.emails = emails
        self.__extra_from_email()  # random the email

    def __extra_from_email(self):
        random.shuffle(self.emails)
        self.content = [email.vector for email in self.emails]
        self.label = list()
        for email in self.emails:
            if email.is_spam:
                self.label.append([0, 1, 0])
            else:
                self.label.append([1, 0, 0])

    def next_batch(self, batch_size):
        if batch_size + self.index >= len(self.content):
            remain = batch_size + self.index - len(self.content)
            content_result = list(self.content[self.index:])
            label_result = list(self.label[self.index:])
            self.__extra_from_email()  # random the email
            self.index = remain
            content_result.extend(self.content[:self.index])
            label_result.extend(self.label[:self.index])
            return self.__concatenate(content_result), label_result

        else:
            start = self.index
            self.index += batch_size
            return self.__concatenate(self.content[start:(start + batch_size)]), list(
                self.label[start:(start + batch_size)])

    def __concatenate(self, content_list: list):  # 10*16*32
        result = list()
        for content in content_list:
            result.append(np.concatenate([c for c in content], axis=0))
        return result


# construct the model
X_dim = embedding_size * N_WORDS
Z_dim = 100  # noise dimension
H_dim = 128  # hidden dimension
batch_size = 10  # batch size


# initializer
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Example generator
def sample_Z(m, n):
    radius = 0.1
    return np.random.uniform(-radius, radius, size=[m, n])


# Discriminator Net


with tf.name_scope('D_input'):
    X = tf.placeholder(tf.float32, shape=[batch_size, X_dim], name='X')
    Y = tf.placeholder(tf.float32, shape=[batch_size, 3], name="Y")

with tf.name_scope('discriminator'):
    D_W1 = tf.Variable(xavier_init([X_dim, H_dim]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[H_dim]), name='D_b1')

    D_W2 = tf.Variable(xavier_init([H_dim, 3]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[3]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
with tf.name_scope('G_input'):
    Z = tf.placeholder(tf.float32, shape=[batch_size, Z_dim], name='Z')

with tf.name_scope('generator'):
    G_W1 = tf.Variable(xavier_init([Z_dim, H_dim]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[H_dim]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([H_dim, X_dim]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    with tf.name_scope('generator'):
        # z = tf.Print(z,data=[z],summarize=20,first_n=3)
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        # G_h1 = tf.Print(G_h1,data=[G_W1],summarize=20)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_example = G_log_prob

        # G_example = tf.Print(G_example,data=[G_example],summarize=batch_size*N_WORDS*embedding_size,first_n=3)

        return G_example


def discriminator(x):
    with tf.name_scope('discriminator'):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.softmax(D_logit)
        # D_prob = tf.Print(D_prob, data=[], summarize=20)

        return D_prob, D_logit


with tf.name_scope('D_loss'):
    G_sample = generator(Z)
    # G_sample = tf.Print(G_sample,data=[G_sample[:1,64:128]],summarize=20,message="sample:")
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)
    D_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=D_logit_real))
    D_loss = tf.reduce_mean(D_real_loss + (1. - D_fake[:, -1]))
    # D_loss = tf.Print(D_loss, data=[D_real], summarize=20, message="D_real:")
    # D_loss = tf.Print(D_loss, data=[D_fake], summarize=20, message="D_fake:")
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.histogram("D_logit_fake", D_logit_fake)
    tf.summary.histogram("Y", Y)
    tf.summary.histogram("D_real", D_real[:, :-1])
    # D_loss = tf.Print(D_loss, data=[Y], summarize=20, message="Y")
    # D_loss = tf.Print(D_loss, data=[D_real], summarize=20, message="D_real")
    # D_loss = tf.Print(D_loss,data=[D_loss],summarize=20)

with tf.name_scope('G_loss'):
    G_loss = tf.reduce_mean(D_fake[:, -1])
    # G_loss = tf.Print(G_loss, data=[G_loss], summarize=20, message="G:")
    tf.summary.scalar("G_loss", G_loss)

# Only update D(X)'s parameters, so var_list = theta_D
with tf.name_scope('D_train'):
    D_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
with tf.name_scope('G_train'):
    G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=theta_G)

with tf.name_scope('evaluate'):
    X_eval = tf.placeholder(tf.float32, shape=[batch_size, X_dim], name='X_eval')
    Y_eval = tf.placeholder(tf.float32, shape=[batch_size, 3], name="Y_eval")
    D_result, D_logit_result = discriminator(X_eval)
    result = tf.argmax(D_result, 1)  # 1 for ham,0 for spam
    # result = tf.Print(result,data=[result],summarize=20)
    expect = tf.cast(Y_eval[:, 1], dtype=tf.int64)
    evaluator = tf.reduce_sum(tf.abs(result - expect)) / float(batch_size)
    # evaluator = tf.Print(evaluator,data=[evaluator],summarize=20)

i = 0
dictionary, word_embeddings = read_data()
emails = extract_email(dictionary, word_embeddings, N_WORDS, SOURCE_MSG, SOURCE_SPAM)
test_emails = extract_email(dictionary, word_embeddings, N_WORDS, TEST_MSG, TEST_SPAM)
batch_manager = BatchManager(emails)
test_batch_manager = BatchManager(test_emails)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/spam-filter")
graph_writer = tf.summary.FileWriter("log/spam-filter", sess.graph)

evaluator_result = []

for it in range(5000):
    x_mb, y_mb = batch_manager.next_batch(batch_size)
    _, D_loss_curr, summary = sess.run([D_solver, D_loss, merged],
                                       feed_dict={X: x_mb, Y: y_mb, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    # # log
    # writer.add_summary(summary, it)



    if it % 1000 == 0:
        # print('Iter: {}'.format(it))
        # print('D loss: {:.4}'.format(D_loss_curr))
        # print('G_loss: {:.4}'.format(G_loss_curr))
        # print()
        x_mb, y_mb = test_batch_manager.next_batch(batch_size)
        evaluator_curr = sess.run([evaluator], feed_dict={X_eval: x_mb, Y_eval: y_mb})
        evaluator_result.append(evaluator_curr[0])

print(sum(evaluator_result) / len(evaluator_result))
