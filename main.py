from mxnet import gluon
from mxnet import nd

from options import Options
from trainer import Trainer

import mxnet as mx
import numpy as np


def get_transformer(opts):
    def transformer(data, label):
        data = mx.image.imresize(data, opts.img_size, opts.img_size)
        data = nd.transpose(data, (2, 0, 1))
        data = (data.astype(np.float32) / 128.0) - 1.0
        if data.shape[0] == 1:
            data = nd.tile(data, (3, 1, 1))
        return data, label
    return transformer


def get_dataset_from_folder(opts):
    func = get_transformer(opts)
    train_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(opts.data_root, transform=func),
                                       batch_size=opts.batch_size, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(opts.data_root, transform=func),
                                      batch_size=opts.batch_size, shuffle=False, last_batch='discard')

    return train_data, test_data


if __name__ == '__main__':
    options = Options(64, '/home/wallart/datasets/lfw-deepfunneled', '/home/wallart/models')
    train_dataset, _ = get_dataset_from_folder(options)
    trainer = Trainer(options)
    trainer.train(train_dataset)
