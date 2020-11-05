from dataset.cifar10 import Cifar10
from dataset.image_folder import ImageFolder
from dataset.mnist import Mnist

datasets = {
    'cifar10': Cifar10,
    'mnist': Mnist,
    'folder': ImageFolder,
}