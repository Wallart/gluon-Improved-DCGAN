from abc import ABC, abstractmethod
from mxnet.gluon import data
from mxnet import image

import numpy as np


class DatasetException(Exception):
    pass


class AbstractDataset(ABC):
    def __init__(self, opts, shuffle=True):
        self._opts = opts
        self._conf = {
            'batch_size': opts.batch_size,
            'shuffle': shuffle,
            'last_batch': 'discard',
            'num_workers': opts.workers
        }
        self._dataset = self._init_dataset(opts.path)

    def get(self):
        return data.DataLoader(self._dataset, **self._conf)

    @abstractmethod
    def _init_dataset(self, path):
        pass

    @staticmethod
    def get_transformer(opts):
        def transformer(images, label):
            images = image.imresize(images, opts.image_size, opts.image_size)
            images = images.transpose((2, 0, 1))
            images = (images.astype(np.float32) / 127.5) - 1.0

            return images, label

        return transformer
