from dataset.abstract_dataset import AbstractDataset, DatasetException
from mxnet.gluon import data


class ImageFolder(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)

    def _init_dataset(self, path=None):
        if not path:
            raise DatasetException('Missing dataset path')

        func = ImageFolder.get_transformer(self._opts)
        colored = 1 if self._opts.num_colors == 3 else 0
        return data.vision.ImageFolderDataset(path, transform=func, flag=colored)
