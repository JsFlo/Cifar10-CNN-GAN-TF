from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cnn_cifar10_trainer as trainer_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

IMAGE_SIZE = 32

TRAINING_STEPS = 5000
SAMPLE_FREQUENCY = 50
DEBUG_PRINT_FREQUENCY = 10000
SAMPLE_WIDTH = 3
SAMPLE_HEIGHT = 3
NUMBER_OF_SAMPLES = SAMPLE_WIDTH * SAMPLE_HEIGHT
FILE_SAMPLE_OUTPUT_PATH = "out/"
BATCH_SIZE = 128


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(SAMPLE_WIDTH, SAMPLE_HEIGHT))
    gs = gridspec.GridSpec(SAMPLE_WIDTH, SAMPLE_HEIGHT)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # convert 784 to 28 x 28
        plt.imshow(sample.reshape(32, 32, 3), cmap='Greys_r')

    return fig


def printShape(string, tensor):
    print(string)
    print(tensor.shape)


def variable(name, shape, initializer):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def pool(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)


def pool3(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1],
                          padding='SAME', name=name)


def pool4(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1],
                          padding='SAME', name=name)


my_weights = {
    'wc1': _variable_with_weight_decay('weightsc1',
                                       shape=[5, 5, 3, 64],
                                       stddev=5e-2,
                                       wd=0.0),
    'wc2': _variable_with_weight_decay('weightsc2',
                                       shape=[5, 5, 64, 64],
                                       stddev=5e-2,
                                       wd=0.0),
    'wl3': _variable_with_weight_decay('weightsl3',
                                       shape=[4096, 384],
                                       stddev=0.04,
                                       wd=0.004),
    'wl4': _variable_with_weight_decay('weightsl4', shape=[384, 192],
                                       stddev=0.04, wd=0.004),
    'w5': _variable_with_weight_decay('w5', [192, 10],
                                      stddev=1 / 192.0, wd=0.0),
    'w6': _variable_with_weight_decay('w6', shape=[10, 200],
                                      stddev=0.04, wd=0.004),
    'w7': _variable_with_weight_decay('w7', shape=[200, 2],
                                      stddev=1 / 200, wd=0.000)
}

my_biases = {
    'bc1': variable('biasesc1', [64], tf.constant_initializer(0.0)),
    'bc2': variable('biasesc2', [64], tf.constant_initializer(0.1)),
    'bl3': variable('biasesc3', [384], tf.constant_initializer(0.1)),
    'bl4': variable('biasesl4', [192], tf.constant_initializer(0.1)),
    'b5': variable('b5', [10], tf.constant_initializer(0.0)),
    'b6': variable('b6', [200], tf.constant_initializer(0.1)),
    'b7': variable('b7', [2], tf.constant_initializer(0.0)),
}

generator_w = {
    'gw1': _variable_with_weight_decay('gw1',
                                       shape=[5, 5, 1, 64],
                                       stddev=5e-2,
                                       wd=0.0),
    'gw2': _variable_with_weight_decay('gw2',
                                       shape=[5, 5, 64, 128],
                                       stddev=5e-2,
                                       wd=0.0),
    'gw3': _variable_with_weight_decay('gw3',
                                       shape=[3, 3, 128, 256],
                                       stddev=0.04,
                                       wd=0.004),
    'gw4': _variable_with_weight_decay('gw4',
                                       shape=[256 * 5 * 5, 4096],
                                       stddev=0.04,
                                       wd=0.004),
    'gw5': _variable_with_weight_decay('gw5',
                                       [4096, 3200],
                                       stddev=1 / 4096.0,
                                       wd=0.0),
    'gw6': _variable_with_weight_decay('gw6',
                                       shape=[3200, 3172],
                                       stddev=0.04,
                                       wd=0.004),
    'gw7': _variable_with_weight_decay('gw7',
                                       shape=[3172, 3072],
                                       stddev=1 / 4096,
                                       wd=0.000)
}

generator_b = {
    'gb1': variable('gb1', [64], tf.constant_initializer(0.0)),
    'gb2': variable('gb2', [128], tf.constant_initializer(0.1)),
    'gb3': variable('gb3', [256], tf.constant_initializer(0.1)),
    'gb4': variable('gb4', [4096], tf.constant_initializer(0.1)),
    'gb5': variable('gb5', [3200], tf.constant_initializer(0.0)),
    'gb6': variable('gb6', [3172], tf.constant_initializer(0.1)),
    'gb7': variable('gb7', [3072], tf.constant_initializer(0.0)),
}


def generator(image):
    # conv1
    conv1 = tf.nn.conv2d(image, generator_w['gw1'], [1, 1, 1, 1], padding='SAME')
    conv1_pre_activation = tf.nn.bias_add(conv1, generator_b['gb1'])
    conv1_activated = tf.nn.relu(conv1_pre_activation)
    printShape("conv1", conv1_activated)
    conv1_pool1 = pool(conv1_activated, 'pool1')
    printShape("conv1_pool1", conv1_pool1)
    conv1_norm1 = tf.nn.lrn(conv1_pool1,
                            4,
                            bias=1.0,
                            alpha=0.001 / 9.0,
                            beta=0.75,
                            name='norm1')

    # conv2
    conv2 = tf.nn.conv2d(conv1_norm1, generator_w['gw2'], [1, 1, 1, 1], padding='SAME')
    conv2_pre_activation = tf.nn.bias_add(conv2, generator_b['gb2'])
    conv2_activated = tf.nn.relu(conv2_pre_activation)
    printShape("conv2", conv2_activated)
    conv2_norm1 = tf.nn.lrn(conv2_activated,
                            4,
                            bias=1.0,
                            alpha=0.001 / 9.0,
                            beta=0.75,
                            name='norm2')
    conv2_pool1 = pool3(conv2_norm1, 'pool2')
    printShape("conv2_pool2", conv2_pool1)

    # conv2
    conv3 = tf.nn.conv2d(conv2_pool1, generator_w['gw3'], [1, 1, 1, 1], padding='SAME')
    conv3_pre_activation = tf.nn.bias_add(conv3, generator_b['gb3'])
    conv3_activated = tf.nn.relu(conv3_pre_activation)
    printShape("conv3", conv3_activated)
    conv3_norm1 = tf.nn.lrn(conv3_activated,
                            4,
                            bias=1.0,
                            alpha=0.001 / 9.0,
                            beta=0.75,
                            name='norm3')
    conv3_pool1 = pool4(conv3_norm1, 'pool3')
    printShape("conv3_pool1", conv3_pool1)

    # local4
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(conv3_pool1, [-1, 256 * 5 * 5])
    printShape("reshape", reshape)

    local4 = tf.nn.relu(tf.matmul(reshape, generator_w['gw4']) + generator_b['gb4'])
    printShape("fc4", local4)

    # local5
    local5 = tf.nn.relu(tf.matmul(local4, generator_w['gw5']) + generator_b['gb5'])
    printShape("local5", local5)

    # local6
    local6 = tf.nn.relu(tf.matmul(local5, generator_w['gw6']) + generator_b['gb6'])
    printShape("local6", local6)

    softmax_linear = tf.add(tf.matmul(local6, generator_w['gw7']), generator_b['gb7'])
    printShape("final", softmax_linear)

    generated_reshaped = tf.reshape(softmax_linear, [-1, 32, 32, 3])
    printShape("reshapedOut", generated_reshaped)

    logit = generated_reshaped
    prob = tf.nn.sigmoid(logit)
    return prob, logit


def discriminator(image):
    # conv1
    conv = tf.nn.conv2d(image, my_weights['wc1'], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, my_biases['bc1'])
    conv1 = tf.nn.relu(pre_activation)
    printShape("conv1", conv1)

    # pool1
    pool1 = pool(conv1, 'pool1')
    printShape("pool1", pool1)
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    printShape("norm1", norm1)

    # conv2
    conv = tf.nn.conv2d(norm1, my_weights['wc2'], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, my_biases['bc2'])
    conv2 = tf.nn.relu(pre_activation)
    printShape("conv2", conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    printShape("norm2", norm2)
    # pool2
    pool2 = pool(norm2, 'pool2')
    printShape("pool2", pool2)

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [-1, 4096])
    printShape("reshape", reshape)
    local3 = tf.nn.relu(tf.matmul(reshape, my_weights['wl3']) + my_biases['bl3'])
    printShape("local3", local3)

    # local4
    local4 = tf.nn.relu(tf.matmul(local3, my_weights['wl4']) + my_biases['bl4'])
    printShape("local4", local4)

    intermediate_out_5 = tf.add(tf.matmul(local4, my_weights['w5']), my_biases['b5'])
    printShape("intermediate", intermediate_out_5)

    local6 = tf.nn.relu(tf.matmul(intermediate_out_5, my_weights['w6']) + my_biases['b6'])
    printShape("local6", local6)

    softmax_linear = tf.add(tf.matmul(local6, my_weights['w7']), my_biases['b7'])
    printShape("softmax", softmax_linear)

    logit = softmax_linear
    prob = tf.nn.sigmoid(logit)
    return prob, logit


def get_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def pretty_loss(loss_values):
    plt.plot([np.mean(loss_values[i]) for i in range(len(loss_values))])
    plt.show()
    plt.savefig("loss_over_time.png")
    plt.close()


def pretty_accuracy(accuracy):
    plt.plot([np.mean(accuracy[i]) for i in range(len(accuracy))])
    plt.show()
    plt.savefig("accuracy_over_time.png")
    plt.close()


def pretty_loss_avg(loss_values, avg):
    plt.plot([np.mean(loss_values[i - avg:i]) for i in range(len(loss_values))])
    plt.show()
    plt.savefig("loss_over_time_avg.png")
    plt.close()


def pretty_accuracy_avg(accuracy, avg):
    plt.plot([np.mean(accuracy[i - avg:i]) for i in range(len(accuracy))])
    plt.show()
    plt.savefig("accuracy_over_time_avg.png")
    plt.close()


def print_loss_accuracy(loss_values, accuracy):
    pretty_loss_avg(loss_values, 50)
    pretty_loss(loss_values)
    pretty_accuracy(accuracy)
    pretty_accuracy_avg(accuracy, 3)


def main(argv=None):  # pylint: disable=unused-argument
    # trainer = trainer_input.Trainer(50)
    # image_placeholder = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])
    # generated_out = generator(image_placeholder)
    # discriminator(generated_out)

    # going to be randomly generated
    numberOfInputs = 10000
    # input to the generator of 100 numbers of noise
    Z = tf.placeholder(tf.float32, shape=[None, numberOfInputs])
    Z_reshaped = tf.reshape(Z, [-1, 100, 100, 1])

    # generator will take in Z (random noise of 100) and output an image that's 28 x 28
    G_sample, _ = generator(Z_reshaped)

    # this discriminator will take in the real images
    x_image = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    D_real, D_logit_real = discriminator(x_image)

    # discriminator will take in the fake images the generator generates
    D_fake, D_logit_fake = discriminator(G_sample)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    # Only update D(X)'s parameters, so var_list = theta_D
    theta_D = [my_weights['wc1'], my_weights['wc2'], my_weights['wl3'],
               my_weights['wl4'], my_weights['w5'], my_weights['w6'], my_weights['w7'],
               my_biases['bc1'], my_biases['bc2'], my_biases['bl3'],
               my_biases['bl4'], my_biases['b5'], my_biases['b6'], my_biases['b7']]

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

    # Only update G(X)'s parameters, so var_list = theta_G
    theta_G = [generator_w['gw1'], generator_w['gw2'], generator_w['gw3'],
               generator_w['gw4'], generator_w['gw5'], generator_w['gw6'], generator_w['gw7'],
               generator_b['gb1'], generator_b['gb2'], generator_b['gb3'],
               generator_b['gb4'], generator_b['gb5'], generator_b['gb6'], generator_b['gb7']]

    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    trainer = trainer_input.Trainer(50)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for it in range(TRAINING_STEPS):

        train_batch, labels = trainer.next_batch()

        if it % SAMPLE_FREQUENCY == 0:
            # sample the generator and save the plots
            samples = sess.run(G_sample,
                               feed_dict={Z: sample_Z(NUMBER_OF_SAMPLES, numberOfInputs)})
            fig = plot(samples)
            plt.savefig('{}{}.png'.format(FILE_SAMPLE_OUTPUT_PATH, str(it).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={x_image: train_batch, Z: sample_Z(BATCH_SIZE, numberOfInputs)})
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={Z: sample_Z(BATCH_SIZE, numberOfInputs)})

        if it % DEBUG_PRINT_FREQUENCY == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
    print()


main()
