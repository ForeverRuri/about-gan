import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 32
noise_dim = 20
image_dim  = 784
learning_rate = 0.001
hidden_size = 100

def glorot_init(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.)))

g = {'weights_1' : glorot_init([noise_dim, hidden_size]),
     'weights_2' : glorot_init([hidden_size, image_dim]),
     'bias_1' : tf.Variable(tf.zeros([hidden_size])),
     'bias_2' : tf.Variable(tf.zeros([image_dim]))}

d = {'weights_1' : glorot_init([image_dim, hidden_size]),
     'weights_2' : glorot_init([hidden_size, 1]),
     'bias_1' : tf.Variable(tf.zeros([hidden_size])),
     'bias_2' : tf.Variable(tf.zeros([1]))}

def generate(x):
    out = tf.matmul(x, g['weights_1'])
    out = tf.add(out, g['bias_1'])
    out = tf.nn.relu(out)    
    out = tf.matmul(out, g['weights_2'])
    out = tf.add(out, g['bias_2'])
    out = tf.nn.sigmoid(out)
    return out

def discriminate(x):
    out = tf.matmul(x, d['weights_1'])
    out = tf.add(out, d['bias_1'])
    out = tf.nn.relu(out)    
    out = tf.matmul(out, d['weights_2'])
    out = tf.add(out, d['bias_2'])
    out = tf.nn.sigmoid(out)
    return out

noise = tf.placeholder(tf.float32, [None, noise_dim])
disc_input = tf.placeholder(tf.float32, [None, image_dim])

gen_sample = generate(noise)
disc_real = discriminate(disc_input)
disc_fake = discriminate(gen_sample)

gen_loss = - tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(1. - disc_fake) + tf.log(disc_real))

gen_vars = [g['weights_1'], g['weights_2'],
            g['bias_1'], g['bias_2']]
disc_vars = [d['weights_1'], d['weights_2'],
            d['bias_1'], d['bias_2']]


gen_train = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list = gen_vars)
disc_train = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list = disc_vars)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

num_steps = 20000
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = mnist.train.next_batch(batch_size)
    # Generate noise to feed to the generator
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

    # Train
    feed_dict = {disc_input: batch_x, noise: z}
    _, _, gl, dl = sess.run([gen_train, disc_train, gen_loss, disc_loss],
                            feed_dict=feed_dict)
    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))


# 尝试来一波输出
# 10行，每一行6个生成的数字
n_show = 10
each_n = 6
canvas = np.empty([n_show * 28, 28 * each_n])
for i in range(n_show):
    show_noise = np.random.uniform(-1, 1. ,size = [each_n, noise_dim])
    digit = sess.run(gen_sample, feed_dict = {noise : show_noise})
    digit = -1 * (digit - 1)
    for j in range(each_n):
        canvas[i * 28: (i+1) * 28, j * 28 : (j+1) * 28] = digit[j].reshape([28, 28])
plt.figure()
plt.imshow(canvas) 
plt.show()