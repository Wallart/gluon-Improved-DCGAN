from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from datetime import datetime
from discriminator import Discriminator
from generator import Generator

import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import time
import logging


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class Trainer:

    def __init__(self, epochs, batch_size, img_size, ctx, g_lr, d_lr, latent_z_size, weight_decay):
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.ctx = ctx

        self.outdir = '/home/wallart/Desktop/'
        self.dataset = 'cifar10'

        self.g_lr = g_lr
        self.d_lr = d_lr
        self.latent_z_size = latent_z_size
        self.wd = weight_decay

        self.d = Discriminator(img_size=img_size)
        self.g = Generator(img_size=img_size, z_size=latent_z_size)

        self.stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
        logging.basicConfig(level=logging.DEBUG)

    def visualize(self, img_arr):
        plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.axis('off')

    def train(self, train_data):

        real_label = nd.ones((self.batch_size,), ctx=self.ctx)
        fake_label = nd.zeros((self.batch_size,), ctx=self.ctx)

        metric = mx.metric.CustomMetric(facc)

        self.g.initialize()
        self.d.initialize()

        g_params = {'learning_rate': self.g_lr, 'wd': self.wd, 'beta1': 0.5, 'beta2': 0.999}
        d_params = {'learning_rate': self.d_lr, 'wd': self.wd, 'beta1': 0.5, 'beta2': 0.999}

        g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', g_params)
        d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', d_params)

        loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

        for epoch in range(self.epochs):
            fake = None
            tic = time.time()
            btic = time.time()

            for i, (d, l) in enumerate(train_data):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                data = d.as_in_context(self.ctx)
                latent_z = mx.nd.random_normal(0, 1, shape=(self.batch_size, self.latent_z_size, 1, 1), ctx=self.ctx)

                with autograd.record():
                    # Train with real image
                    output = self.d(data).reshape((-1, 1))
                    err_d_real = loss(output, real_label)
                    metric.update([real_label, ], [output, ])

                    # Train with fake image
                    fake = self.g(latent_z)
                    output = self.d(fake.detach()).reshape((-1, 1))
                    err_d_fake = loss(output, fake_label)
                    err_d = err_d_real + err_d_fake
                    err_d.backward()
                    metric.update([fake_label, ], [output, ])

                d_trainer.step(self.batch_size)

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                with autograd.record():
                    fake = self.g(latent_z)
                    output = self.d(fake).reshape((-1, 1))
                    err_g = loss(output, real_label)
                    err_g.backward()

                g_trainer.step(self.batch_size)

                # Print log info every 10 batches
                if i % 10 == 0:
                    name, acc = metric.get()
                    logging.info('Speed: {} samples/s'.format(self.batch_size / (time.time() - btic)))
                    logging.info(
                        'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                        % (nd.mean(err_d).asscalar(),
                           nd.mean(err_g).asscalar(), acc, i, epoch))
                btic = time.time()

            name, acc = metric.get()
            metric.reset()

            logging.info('\nbinary training acc at epoch {}: {}={}'.format(epoch, name, acc))
            logging.info('time: {}'.format(time.time() - tic))

            # Visualize one generated image for each epoch
            fake_img = fake[0]
            self.visualize(fake_img)
            plt.show()
