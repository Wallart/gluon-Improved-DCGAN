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

        mult = self.opts.image_size // 8

        with self.name_scope():
            self.stages = nn.HybridSequential()

            self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size * mult, 4, strides=1, padding=0, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.Activation('relu'))

            while mult > 1:
                self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size * (mult // 2), 4, strides=2, padding=1, use_bias=False, **self.init))
                if self.opts.with_selu:
                    self.stages.add(nn.SELU())
                else:
                    self.stages.add(nn.BatchNorm())
                    self.stages.add(nn.Activation('relu'))
                mult = mult // 2

            # End block
            self.stages.add(nn.Conv2DTranspose(self.opts.num_colors, 4, strides=2, padding=1, use_bias=False, **self.init))
            self.stages.add(nn.Activation('tanh'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.stages(x)
