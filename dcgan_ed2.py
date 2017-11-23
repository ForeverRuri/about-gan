import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def leakyReLu(x, alpha = 0.1):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)



noise_dim = 100
batch_size = 32
num_steps = 10000
lr_generator = 0.002
lr_discriminator = 0.002

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
noise_dim = 100 # Noise data points# Build Networks


noise = tf.placeholder(tf.float32, [None, noise_dim])
real_image = tf.placeholder(tf.float32, [None, 28, 28, 1])

def discriminator(x, reuse = False):
    with tf.variable_scope('discriminator', reuse = reuse):
        # out 28*28*1
        out = tf.layers.conv2d(x, 64, 5, 2, 'same')
        # out 14*14*64
        out = leakyReLu(out)
        out = tf.layers.conv2d(out, 128, 5, 2, 'same')
        # out 7*7*128
        out = leakyReLu(out)
        out = tf.reshape(out, [-1, 7*7*128])
        out = tf.layers.dense(out, 2)
        return out

def generator(x, reuse = False):
    with tf.variable_scope('generator', reuse = reuse):
        out = tf.layers.dense(x, 7*7*128)
        out = tf.reshape(out, [-1, 7, 7, 128])
        # 7*7*128
        out = tf.layers.conv2d_transpose(out, 64, 5, 2, 'same')
        # 14*14*64
        out = tf.layers.conv2d_transpose(out, 1, 5, 2, 'same')
        # 28*28*1
        out = tf.nn.tanh(out)
        return out

gen_samples = generator(noise)

disc_real = discriminator(real_image)
# inference时reuse为True
# 所有的参数进行复用
disc_fake = discriminator(gen_samples, reuse=True)

stacked_gan = discriminator(gen_samples, reuse=True)



disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))

disc_loss = disc_loss_real + disc_loss_fake

gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits = disc_fake, labels = tf.ones([batch_size], dtype = tf.int32)))

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')

optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
# `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, 1 + num_steps):
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
    # Rescale to [-1, 1], the input range of the discriminator
    batch_x = batch_x * 2. - 1.
    # 训练判别器
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, dl = sess.run([train_disc, disc_loss], feed_dict={real_image: batch_x, noise: z})
    # 训练生成器
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, gl = sess.run([train_gen, gen_loss], feed_dict={noise: z})
    
    if i % 50 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # Generate image from noise.
    g = sess.run(gen_samples, feed_dict={noise: z})
    # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
    g = (g + 1.) / 2.
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()