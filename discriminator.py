from mxnet import gluon
from mxnet.gluon import nn

import mxnet as mx


class Discriminator(gluon.HybridBlock):

    def __init__(self, img_size=64, hidden_size=128, n_colors=3, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.img_size = img_size
        self.h_size = hidden_size
        self.n_colors = n_colors

        self.init = {
            'weight_initializer': mx.init.Normal(0.02)
        }

        with self.name_scope():
            self.stages = nn.HybridSequential()

            self.stages.add(nn.Conv2D(self.h_size, 4, strides=2, padding=1, use_bias=False, **self.init))
            self.stages.add(nn.SELU())

            new_img_size = img_size // 2
            mult = 1

            while new_img_size > 4:
                self.stages.add(nn.Conv2D(self.h_size * (2 * mult), 4, strides=2, padding=1, use_bias=False, **self.init))
                self.stages.add(nn.SELU())

                new_img_size = new_img_size // 2
                mult *= 2

            # End block
            self.stages.add(nn.Conv2D(1, 4, strides=1, padding=0, use_bias=False, **self.init))
            self.stages.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.stages(x).reshape(32, -1)  # .view(-1) in PyTorch => we need to rearrange 1x1x1 to 1
