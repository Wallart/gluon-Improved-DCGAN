from dataset.abstract_dataset import AbstractDataset
from mxnet.gluon import data


class Mnist(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(Mnist, self).__init__(*args, **kwargs)

    def _init_dataset(self, path=None):
        opts = {
            'train': True,
            'transform': Mnist.get_transformer(self._opts)
        }
        if path:
            opts['root'] = path
        return data.vision.MNIST(**opts)
