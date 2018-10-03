from mxnet import gluon
from mxnet.gluon import nn

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


class Discriminator(gluon.HybridBlock):

    def __init__(self, opts, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.opts = opts
        self.init = {
            'weight_initializer': mx.init.Normal(0.02)
        }

        i = 0
        mult = 1
        feature_map_size = self.opts.image_size

        with self.name_scope():
            self.stages = nn.HybridSequential()
            # We have to produce (? x 4 x 4) features maps at layer n-1, whatever the original image size was
            while feature_map_size > 4:
                layer = nn.HybridSequential()
                layer.add(nn.Conv2D(self.opts.ndf * mult, 4, 2, 1, use_bias=False, **self.init))
                if i > 0:
                    layer.add(nn.BatchNorm())
                if self.opts.relu:
                    layer.add(nn.LeakyReLU(0.2))
                else:
                    layer.add(nn.SELU())

                feature_map_size = feature_map_size // 2
                i += 1
                mult *= 2
                self.stages.add(layer)

            layer = nn.HybridSequential(prefix='')
            layer.add(nn.Conv2D(1, 4, 1, 0, use_bias=False, **self.init))
            layer.add(nn.Activation('sigmoid'))
            self.stages.add(layer)

            assert len(self.stages) == N_LAYERS[self.opts.image_size]
            assert self.stages[0][-2]._channels == self.opts.ndf
            assert self.stages[-2][-3]._channels == self.opts.ndf * self.opts.image_size // 8

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.stages(x)
        return F.reshape(x, shape=-1)  # .view(-1) in PyTorch => we need to rearrange 1x1x1 to 1
