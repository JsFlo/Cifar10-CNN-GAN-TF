import numpy as np
import pickle
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import sys
import tarfile

from six.moves import urllib

# import download
# from dataset import one_hot_encoded

DEFAULT_DATA_PATH = "data/CIFAR-10/"

# URL for the data-set on the internet.
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# data details
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
# Length of an image when flattened to a 1-dim array.
IMG_SIZE_FLAT = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS
NUM_CLASSES = 10

# TRAIN
TRAIN_NUM_FILES = 5
TRAIN_IMG_PER_FILE = 10000
TRAIN_TOTAL_NUM_IMG = TRAIN_NUM_FILES * TRAIN_IMG_PER_FILE


def _unpickle(file):
    import pickle
    print("Extracting data: " + file)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _convert_images(raw_images):
    """
    Convert images from the CIFAR-10 format
    return 4-dim array. shape: [image_number, height, width, channel]
    """

    # Convert the images from ints to floating-points
    raw_float = np.array(raw_images, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_images_labels(filename):
    raw_data = _unpickle(filename)

    # 10,000 * 3072 ( 32 x 32 x 3 )
    flattened_images = raw_data[b'data']

    labels = raw_data[b'labels']

    # convert list to array
    labels = np.array(labels)

    labels = one_hot_encoded(labels, NUM_CLASSES)

    # Convert the images.
    images = _convert_images(flattened_images)

    return images, labels


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = DEFAULT_DATA_PATH
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_label_names():
    label_names = _unpickle(filename="cifar_batches/batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in label_names]

    return names


def one_hot_encoded(class_numbers, num_classes):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    return np.eye(num_classes, dtype=float)[class_numbers]


# def load_training_data():
#     """
#     Load all the training-data for the CIFAR-10 data-set.
#     Returns the images, labels and one-hot encoded labels.
#     """
#     images = np.zeros(shape=[TRAIN_TOTAL_NUM_IMG, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=float)
#     labels = np.zeros(shape=[TRAIN_TOTAL_NUM_IMG], dtype=int)
#
#     # Begin-index for the current batch.
#     begin = 0
#
#     # For each data-file.
#     for i in range(TRAIN_NUM_FILES):
#         # Load the images and class-numbers from the data-file.
#         images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
#
#         # Number of images in this batch.
#         num_images = len(images_batch)
#
#         # End-index for the current batch.
#         end = begin + num_images
#
#         # Store the images into the array.
#         images[begin:end, :] = images_batch
#
#         # Store the class-numbers into the array.
#         cls[begin:end] = cls_batch
#
#         # The begin-index for the next batch is the current end-index.
#         begin = end
#
#     return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)


#
#
# def load_test_data():
#     """
#     Load all the test-data for the CIFAR-10 data-set.
#     Returns the images, class-numbers and one-hot encoded class-labels.
#     """
#
#     images, cls = _load_data(filename="test_batch")
#
#     return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)

