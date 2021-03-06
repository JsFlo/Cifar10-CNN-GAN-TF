from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cnn_cifar10_trainer as trainer_input
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 32
NUM_CLASSES = 10

TRAIN_STEPS = 500000
PRINT_TRAIN_FREQ = 1000


def printShape(tensor):
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


my_weights = {
    'wc1': _variable_with_weight_decay('weightsc1',
                                       shape=[5, 5, 3, 64],
                                       stddev=5e-2,
                                       wd=0.0),
    'wc2': _variable_with_weight_decay('weightsc2',
                                       shape=[5, 5, 64, 64],
                                       stddev=5e-2,
                                       wd=0.0),
    'wl4': _variable_with_weight_decay('weightsl4', shape=[384, 192],
                                       stddev=0.04, wd=0.004),
    'w5': _variable_with_weight_decay('w5', [192, NUM_CLASSES],
                                      stddev=1 / 192.0, wd=0.0),
    'w6': _variable_with_weight_decay('w6', shape=[NUM_CLASSES, 200],
                                      stddev=0.04, wd=0.004),
    'w7': _variable_with_weight_decay('w7', shape=[200, NUM_CLASSES],
                                      stddev=1 / 200, wd=0.000)
}

my_biases = {
    'bc1': variable('biasesc1', [64], tf.constant_initializer(0.0)),
    'bc2': variable('biasesc2', [64], tf.constant_initializer(0.1)),
    'bl3': variable('biasesc3', [384], tf.constant_initializer(0.1)),
    'bl4': variable('biasesl4', [192], tf.constant_initializer(0.1)),
    'b5': variable('b5', [NUM_CLASSES], tf.constant_initializer(0.0)),
    'b6': variable('b6', [200], tf.constant_initializer(0.1)),
    'b7': variable('b7', [NUM_CLASSES], tf.constant_initializer(0.0)),
}


def pool(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)


def buildModel(image):
    # conv1
    conv = tf.nn.conv2d(image, my_weights['wc1'], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, my_biases['bc1'])
    conv1 = tf.nn.relu(pre_activation)
    print("Conv1")
    printShape(conv1)

    # pool1
    pool1 = pool(conv1, 'pool1')
    printShape(pool1)
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    printShape(norm1)

    # conv2
    conv = tf.nn.conv2d(norm1, my_weights['wc2'], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, my_biases['bc2'])
    conv2 = tf.nn.relu(pre_activation)
    print("Conv2")
    printShape(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    printShape(norm2)
    # pool2
    pool2 = pool(norm2, 'pool2')
    printShape(pool2)

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [-1, 4096])
    printShape(reshape)
    dim = reshape.get_shape()[1].value
    wl3 = _variable_with_weight_decay('weightsl3', shape=[dim, 384],
                                      stddev=0.04, wd=0.004)
    local3 = tf.nn.relu(tf.matmul(reshape, wl3) + my_biases['bl3'])
    printShape(local3)

    # local4
    local4 = tf.nn.relu(tf.matmul(local3, my_weights['wl4']) + my_biases['bl4'])
    printShape(local4)

    intermediate_out_5 = tf.add(tf.matmul(local4, my_weights['w5']), my_biases['b5'])
    printShape(intermediate_out_5)

    local6 = tf.nn.relu(tf.matmul(intermediate_out_5, my_weights['w6']) + my_biases['b6'])
    printShape(local6)

    softmax_linear = tf.add(tf.matmul(local6, my_weights['w7']), my_biases['b7'])
    printShape(softmax_linear)

    return softmax_linear


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
    trainer = trainer_input.Trainer(50)
    image_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])

    # 10 hot vectors (0 - 9)
    y_correct_labels = tf.placeholder(tf.float32, shape=[None, 10])
    logits = buildModel(image_placeholder)

    loss = get_loss(logits, y_correct_labels)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_correct_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    # Should start using "summaries"(tensorboard) instead of this
    loss_values = []
    accuracy_values = []
    trained_steps = 0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(TRAIN_STEPS):
            trained_steps = trained_steps + 1
            batch, labels = trainer.next_batch()
            _, loss_value = sess.run([train_step, loss], feed_dict={image_placeholder: batch, y_correct_labels: labels})
            loss_values.append(loss_value)

            if i % PRINT_TRAIN_FREQ == 0:
                print("step: {}".format(i))
                print("loss: {}".format(loss_value))
                accuracy_value = sess.run(accuracy, feed_dict={image_placeholder: batch, y_correct_labels: labels})
                accuracy_values.append(accuracy_value)
                print('test accuracy %g' % accuracy_value)

        validation_images, validation_labels = trainer.test_batch(10000)
        print('test accuracy %g' % accuracy.eval(
            feed_dict={image_placeholder: validation_images, y_correct_labels: validation_labels}))
        print_loss_accuracy(loss_values, accuracy_values)

        print("num of loss_values: {}".format(len(loss_values)))
        print("num of accuracy values: {}".format(len(accuracy_values)))


if __name__ == '__main__':
    tf.app.run()
