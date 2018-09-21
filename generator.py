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

        #mult = self.opts.img_size // 8

        with self.name_scope():
            self.stages = nn.HybridSequential()

            # input is Z, going into a convolution
            self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size * 8, 4, 1, 0, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.Activation('relu'))

            # state size. (self.opts.g_h_size*8) x 4 x 4
            self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size * 4, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.Activation('relu'))

            # state size. (self.opts.g_h_size*8) x 8 x 8
            self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size * 2, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.Activation('relu'))

            # state size. (self.opts.g_h_size*8) x 16 x 16
            self.stages.add(nn.Conv2DTranspose(self.opts.g_h_size, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.Activation('relu'))
                
            # state size. (self.opts.g_h_size*8) x 32 x 32
            self.stages.add(nn.Conv2DTranspose(self.opts.num_colors, 4, 2, 1, use_bias=False, **self.init))
            self.stages.add(nn.Activation('tanh'))
            # state size. (nc) x 64 x 64

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.stages(x)
