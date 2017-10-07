import numpy as np
import cnn_cifar10_input as input


class Trainer:
    def __init__(self, batch_size):
        self.step = 0
        images, labels = input._load_images_labels("cifar_batches/data_batch_1")
        total = len(images)
        self.num_batches = total / batch_size
        self.image_batch = np.split(images, self.num_batches)
        self.label_batch = np.split(labels, self.num_batches)

    def next_batch(self):
        if (self.step >= self.num_batches):
            self.step = 0
        images = self.image_batch[self.step]
        labels = self.label_batch[self.step]
        return images, labels

    def test_batch(self, batch_size):
        images, labels = input._load_images_labels("cifar_batches/test_batch")
        total = len(images)
        images_batch = np.split(images, total / batch_size)
        labels_batch = np.split(labels, total / batch_size)
        return images_batch[0], labels_batch[0]
