from mxnet import gluon
from mxnet.gluon import nn

import mxnet as mx


class Discriminator(gluon.HybridBlock):

    def __init__(self, opts, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.opts = opts
        self.init = {
            'weight_initializer': mx.init.Normal(0.02)
        }

        with self.name_scope():
            self.stages = nn.HybridSequential()

            # input is (nc) x 64 x 64
            self.stages.add(nn.Conv2D(self.opts.d_h_size, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.LeakyReLU(0.2))

            # state size. (self.opts.d_h_size) x 32 x 32
            self.stages.add(nn.Conv2D(self.opts.d_h_size * 2, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.LeakyReLU(0.2))

            # state size. (self.opts.d_h_size) x 16 x 16
            self.stages.add(nn.Conv2D(self.opts.d_h_size * 4, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.LeakyReLU(0.2))

            # state size. (self.opts.d_h_size) x 8 x 8
            self.stages.add(nn.Conv2D(self.opts.d_h_size * 8, 4, 2, 1, use_bias=False, **self.init))
            if self.opts.with_selu:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.SELU())
            else:
                self.stages.add(nn.BatchNorm())
                self.stages.add(nn.LeakyReLU(0.2))

            # state size. (self.opts.d_h_size) x 4 x 4
            self.stages.add(nn.Conv2D(1, 4, 1, 0, use_bias=False, **self.init))
            # Decreases performances
            self.stages.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.reshape(self.stages(x), shape=(32, -1))  # .view(-1) in PyTorch => we need to rearrange 1x1x1 to 1
