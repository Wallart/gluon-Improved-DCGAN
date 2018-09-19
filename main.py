from mxnet import gluon
from mxnet import nd
from trainer import Trainer

import mxnet as mx
import numpy as np


IMG_SIZE = 32
DATA_ROOT = '/home/wallart/Datasets/lfw-deepfunneled'
BATCH_SIZE = 32
EPOCHS = 200
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0
LATENT_Z_SIZE = 100


def transformer(data, label):
    data = mx.image.imresize(data, IMG_SIZE, IMG_SIZE)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32) / (128.0 - 1.0)
    return data, label
    # if data.shape[0] == 1:
    #     data = nd.tile(data, (3, 1, 1))
    # return data.reshape((1,) + data.shape)


# def cifar10():
#     train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(DATA_ROOT, train=True, transform=transformer),
#                                        batch_size=BATCH_SIZE, shuffle=True, last_batch='discard')
#     test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(DATA_ROOT, train=False, transform=transformer),
#                                       batch_size=BATCH_SIZE, shuffle=False, last_batch='discard')
#     return train_data, test_data


def celebA():
    train_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(DATA_ROOT, transform=transformer),
                                       batch_size=BATCH_SIZE, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(DATA_ROOT, transform=transformer),
                                      batch_size=BATCH_SIZE, shuffle=False, last_batch='discard')

    return train_data, test_data


if __name__ == '__main__':
    train_dataset, _ = celebA()
    trainer = Trainer(EPOCHS, BATCH_SIZE, 32, mx.cpu(), G_LEARNING_RATE, D_LEARNING_RATE, LATENT_Z_SIZE, WEIGHT_DECAY)
    trainer.train(train_dataset)
