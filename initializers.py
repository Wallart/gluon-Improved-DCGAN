from mxnet.initializer import register, Initializer
from mxnet import random


@register
class BatchNormInit(Initializer):
    def __init__(self, sigma=0.02):
        super(BatchNormInit, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _init_weight(self, _, arr):
        random.normal(1, self.sigma, out=arr)
