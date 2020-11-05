from dataset.abstract_dataset import AbstractDataset
from mxnet.gluon import data


class Cifar10(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(Cifar10, self).__init__(*args, **kwargs)

    def _init_dataset(self, path=None):
        opts = {
            'train': True,
            'transform': Cifar10.get_transformer(self._opts)
        }
        if path:
            opts['root'] = path
        return data.vision.CIFAR10(**opts)
