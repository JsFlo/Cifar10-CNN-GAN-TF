import numpy as np
import cnn_cifar10_input as input

# TRAIN
TRAIN_NUM_FILES = 5
TRAIN_IMG_PER_FILE = 10000
TRAIN_TOTAL_NUM_IMG = TRAIN_NUM_FILES * TRAIN_IMG_PER_FILE
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
NUM_CLASSES = 10


class Trainer:
    def __init__(self, batch_size):
        input.maybe_download_and_extract()
        self.step = 0
        images, labels = self._get_train_images_labels()
        total = len(images)
        self.num_batches = total / batch_size
        self.image_batch = np.split(images, self.num_batches)
        self.label_batch = np.split(labels, self.num_batches)

    def next_batch(self):
        print("next batch step: ")
        print(self.step)
        if (self.step >= self.num_batches):
            self.step = 0
        images = self.image_batch[self.step]
        labels = self.label_batch[self.step]
        self.step = self.step + 1
        return images, labels

    def test_batch(self, batch_size):
        images, labels = input._load_images_labels("test_batch")
        total = len(images)
        images_batch = np.split(images, total / batch_size)
        labels_batch = np.split(labels, total / batch_size)
        return images_batch[0], labels_batch[0]

    def _get_train_images_labels(self):
        all_images = np.zeros(shape=[TRAIN_TOTAL_NUM_IMG, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=float)
        all_labels = np.zeros(shape=[TRAIN_TOTAL_NUM_IMG, NUM_CLASSES], dtype=int)
        begin = 0
        for i in range(TRAIN_NUM_FILES):
            images_batch, labels_batch = input._load_images_labels(filename="data_batch_" + str(i + 1))
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            all_images[begin:end, :] = images_batch
            all_labels[begin:end] = labels_batch

            begin = end
        return all_images, all_labels
