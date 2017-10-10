from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cnn_cifar10_trainer as trainer_input

IMAGE_SIZE = 32
NUM_CLASSES = 10

TRAIN_STEPS = 5000
PRINT_TRAIN_FREQ = 100


def printShape(tensor):
    print(tensor.shape)


def variable(name, shape, initializer):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def _variable_with_weight_decay(name, shape, stddev, wd):
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


def buildMultiModel(image, keep_prob):
    print("Input Image")
    printShape(image)

    print("row 1")
    # row 1
    row1_conv1 = tf.nn.conv2d(image,
                              _variable_with_weight_decay('row1conv1w',
                                                          shape=[25, 25, 3, 128],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')

    row1_conv1_pre_activation = tf.nn.bias_add(row1_conv1, variable('row1conv1b', [128], tf.constant_initializer(0.0)))
    row1_out = tf.nn.relu(row1_conv1_pre_activation)
    printShape(row1_out)

    print("row2")
    # row 2
    row2_conv1 = tf.nn.conv2d(image,
                              _variable_with_weight_decay('row2conv1w',
                                                          shape=[15, 15, 3, 64],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')
    row2_conv1_pre_activation = tf.nn.bias_add(row2_conv1, variable('row2conv1b', [64], tf.constant_initializer(0.0)))
    row2_conv1_activation = tf.nn.relu(row2_conv1_pre_activation)
    printShape(row2_conv1_activation)

    row2_conv2 = tf.nn.conv2d(row2_conv1_activation,
                              _variable_with_weight_decay('row2conv2w',
                                                          shape=[11, 11, 64, 128],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')
    row2_conv2_pre_activation = tf.nn.bias_add(row2_conv2, variable('row2conv2b', [128], tf.constant_initializer(0.0)))
    row2_out = tf.nn.relu(row2_conv2_pre_activation)
    printShape(row2_out)

    print("row 3")
    # row 3
    row3_conv1 = tf.nn.conv2d(image,
                              _variable_with_weight_decay('row3conv1w',
                                                          shape=[5, 5, 3, 64],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')
    row3_conv1_pre_activation = tf.nn.bias_add(row3_conv1, variable('row3conv1b', [64], tf.constant_initializer(0.0)))
    row3_conv1_activation = tf.nn.relu(row3_conv1_pre_activation)

    printShape(row3_conv1_activation)

    row3_conv2 = tf.nn.conv2d(row3_conv1_activation,
                              _variable_with_weight_decay('row3conv2w',
                                                          shape=[19, 19, 64, 64],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')

    row3_conv2_pre_activation = tf.nn.bias_add(row3_conv2, variable('row3conv2b', [64], tf.constant_initializer(0.0)))
    row3_conv2_activation = tf.nn.relu(row3_conv2_pre_activation)
    printShape(row3_conv2_activation)

    row3_conv3 = tf.nn.conv2d(row3_conv2_activation,
                              _variable_with_weight_decay('row3conv3w',
                                                          shape=[3, 3, 64, 128],
                                                          stddev=5e-2,
                                                          wd=0.0),
                              [1, 1, 1, 1],
                              padding='VALID')
    row3_conv3_pre_activation = tf.nn.bias_add(row3_conv3, variable('row3conv3b', [128], tf.constant_initializer(0.0)))
    row3_out = tf.nn.relu(row3_conv3_pre_activation)
    printShape(row3_out)

    row1_row2_row3 = tf.concat([tf.concat([row1_out, row2_out], 3), row3_out], 3)
    printShape(row1_row2_row3)
    flattened_layer = tf.reshape(row1_row2_row3, [-1, 8 * 8 * 384])
    printShape(flattened_layer)

    # fc1
    dim = flattened_layer.get_shape()[1].value
    fc1_w = _variable_with_weight_decay('fc1w', shape=[dim, 4096],
                                        stddev=0.04, wd=0.004)
    fc1 = tf.nn.relu(tf.matmul(flattened_layer, fc1_w) + variable('fc1b', [4096], tf.constant_initializer(0.1)))
    printShape(fc1)

    # fc2
    fc2_w = _variable_with_weight_decay('fc2w', shape=[4096, 1000],
                                        stddev=0.04, wd=0.004)
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + variable('fc2b', [1000], tf.constant_initializer(0.1)))
    printShape(fc2)

    # fc2 dropout
    fc2_dropout = tf.nn.dropout(fc2, keep_prob)

    # fc3 intermediate out
    fc3_w = _variable_with_weight_decay('fc3w', shape=[1000, 10],
                                        stddev=1 / 1000, wd=0.00001)
    fc3 = tf.nn.relu(tf.matmul(fc2_dropout, fc3_w) + variable('fc3b', [10], tf.constant_initializer(0.0)))
    printShape(fc3)

    # fc4 learn from intermediate
    fc4_w = _variable_with_weight_decay('fc4w', shape=[10, 1000],
                                        stddev=.05, wd=0.0)
    fc4 = tf.nn.relu(tf.matmul(fc3, fc4_w) + variable('fc4b', [1000], tf.constant_initializer(0.1)))
    printShape(fc4)

    fc5_w = _variable_with_weight_decay('fc5w', shape=[1000, 10],
                                        stddev=1 / 1000, wd=0.0)
    fc5 = tf.nn.relu(tf.matmul(fc4, fc5_w) + variable('fc5b', [10], tf.constant_initializer(0.0)))
    printShape(fc5)
    return fc5


def get_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def main(argv=None):
    trainer = trainer_input.Trainer(50)
    image_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    keep_prob = tf.placeholder(tf.float32)
    logits = buildMultiModel(image_placeholder, keep_prob)

    # 10 hot vectors (0 - 9)
    y_correct_labels = tf.placeholder(tf.float32, shape=[None, 10])
    loss = get_loss(logits, y_correct_labels)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_correct_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(TRAIN_STEPS):
            batch, labels = trainer.next_batch()

            train_step.run(feed_dict={image_placeholder: batch, y_correct_labels: labels, keep_prob: 0.5})

            if i % PRINT_TRAIN_FREQ == 0:
                print("step: {}".format(i))
                print('test accuracy %g' % accuracy.eval(
                    feed_dict={image_placeholder: batch, y_correct_labels: labels, keep_prob: 1.0}))

        validation_images, validation_labels = trainer.test_batch(10)
        print('test accuracy %g' % accuracy.eval(
            feed_dict={image_placeholder: validation_images, y_correct_labels: validation_labels, keep_prob: 1.0}))


if __name__ == '__main__':
    tf.app.run()
