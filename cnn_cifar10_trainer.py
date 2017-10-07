import numpy as np
import cnn_cifar10_input as input


class Trainer:
    def __init__(self, batch_size):
        self.step = 0
        images, labels = input._load_images_labels("cifar_batches/data_batch_1")
        total = len(images)
        self.image_batch = np.split(images, total / batch_size)
        self.label_batch = np.split(labels, total / batch_size)

    def next_batch(self):
        images = self.image_batch[self.step]
        labels = self.label_batch[self.step]
        return images, labels

    def test_batch(self):
        return input._load_images_labels("cifar_batches/test_batch")


Trainer(50)
