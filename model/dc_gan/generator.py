from mxnet import gluon
from mxnet.gluon import nn
from initializers import BatchNormInit

import mxnet as mx

# Nb layers determined by image size
N_LAYERS = {
    4: 1,
    8: 2,
    16: 3,
    32: 4,
    64: 5,
    128: 6,
    256: 7,
    512: 8,
    1024: 9
}


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
            'conv': {
                'weight_initializer': mx.init.Normal(0.02)
            },
            'bn': {
                'beta_initializer': mx.init.Zero(),
                'gamma_initializer': BatchNormInit()
            }
        }

        i = 0
        mult = self.opts.image_size // 8

        with self.name_scope():
            if self.opts.conditional:
                self._embed = nn.Embedding(input_dim=self.opts.num_classes, output_dim=self.opts.embed_size)

                self.z_proj = nn.HybridSequential()
                self.z_proj.add(nn.Dense(self.opts.latent_z_size + self.opts.embed_size, flatten=False))
                # self.z_proj.add(nn.Activation('relu'))

            self.stages = nn.HybridSequential()
            # We have to produce (num_color x img_size x img_size) features maps on the last layer
            while mult > 0:
                strides = 1 if i == 0 else 2
                padding = 0 if i == 0 else 1

                layer = nn.HybridSequential(prefix='')
                layer.add(nn.Conv2DTranspose(self.opts.ngf * mult, 4, strides, padding, use_bias=False, **self.init['conv']))
                layer.add(nn.BatchNorm(**self.init['bn']))
                if self.opts.relu:
                    layer.add(nn.Activation('relu'))
                else:
                    layer.add(nn.SELU())

                i += 1
                mult //= 2
                self.stages.add(layer)

            layer = nn.HybridSequential(prefix='')
            layer.add(nn.Conv2DTranspose(self.opts.num_colors, 4, 2, 1, use_bias=False, **self.init['conv']))
            layer.add(nn.Activation('tanh'))
            self.stages.add(layer)

            assert len(self.stages) == N_LAYERS[self.opts.image_size]
            assert self.stages[0][-3]._channels == self.opts.ngf * self.opts.image_size // 8
            assert self.stages[-2][-3]._channels == self.opts.ngf

    def hybrid_forward(self, F, x, *args, **kwargs):
        c, = args
        if c is not None:
            class_embed = self._embed(c).expand_dims(-1).expand_dims(-1)
            x = F.concat(x, class_embed, dim=1)
            x = self.z_proj(x.transpose((0, 2, 3, 1)))
            x = x.transpose((0, 3, 1, 2))

        return self.stages(x)
