from mxnet import gluon
from mxnet.gluon import nn

import mxnet as mx


class Generator(gluon.HybridBlock):

    def __init__(self, opts, *args, **kwargs):
        """
        Constructor
        :param img_size: size of given images
        :param z_size: 100 dimensional uniform distribution (in the paper)
        :param hidden_size: number of hidden nodes in the generator (128 in the paper)
        :param args:
        :param kwargs:
        """
        super(Generator, self).__init__(*args, **kwargs)
        self.opts = opts
        self.init = {
            'weight_initializer': mx.init.Normal(0.02)
        }

        i = 0
        mult = self.opts.image_size // 8

        with self.name_scope():
            self.stages = nn.HybridSequential()
            # We have to produce (? x img_size / 2 x img_size / 2) features maps at layer n-1, whatever the original image size was
            while mult > 0:
                strides = 1 if i == 0 else 2
                padding = 0 if i == 0 else 1

                layer = nn.HybridSequential(prefix='')
                layer.add(nn.Conv2DTranspose(self.opts.g_h_size * mult, 4, strides, padding, use_bias=False, **self.init))
                if self.opts.relu:
                    layer.add(nn.BatchNorm())
                    layer.add(nn.Activation('relu'))
                else:
                    layer.add(nn.BatchNorm())
                    layer.add(nn.SELU())

                i += 1
                mult //= 2
                self.stages.add(layer)

            layer = nn.HybridSequential(prefix='')
            layer.add(nn.Conv2DTranspose(self.opts.num_colors, 4, 2, 1, use_bias=False, **self.init))
            layer.add(nn.Activation('tanh'))
            self.stages.add(layer)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.stages(x)
