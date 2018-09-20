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
import os


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class Trainer:

    def __init__(self, opts):
        self.opts = opts
        self.d = Discriminator(opts)
        self.g = Generator(opts)

        self.stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
        logging.basicConfig(level=logging.DEBUG)

    def visualize(self, img_arr, epoch):
        plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.axis('off')
        if self.opts.visualize:
            plt.show()
        else:
            base_path = os.path.join(self.opts.output_dir, self.stamp, 'images')
            if not os.path.isdir(base_path):
                os.makedirs(base_path)

            plt.savefig(os.path.join(base_path, '{}.png'.format(epoch)))

    def train(self, train_data):

        real_label = nd.ones((self.opts.batch_size,), ctx=self.opts.ctx)
        fake_label = nd.zeros((self.opts.batch_size,), ctx=self.opts.ctx)

        metric = mx.metric.CustomMetric(facc)

        self.g.initialize(ctx=self.opts.ctx)
        self.d.initialize(ctx=self.opts.ctx)

        g_params = {'learning_rate': self.opts.g_lr, 'wd': self.opts.wd, 'beta1': 0.5, 'beta2': 0.999}
        d_params = {'learning_rate': self.opts.d_lr, 'wd': self.opts.wd, 'beta1': 0.5, 'beta2': 0.999}

        g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', g_params)
        d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', d_params)

        loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

        for epoch in range(self.opts.epochs):
            fake = None
            tic = time.time()
            btic = time.time()

            for i, (d, l) in enumerate(train_data):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                data = d.as_in_context(self.opts.ctx)
                latent_z = nd.random_normal(0, 1, shape=(self.opts.batch_size, self.opts.latent_z_size, 1, 1), ctx=self.opts.ctx)

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

                d_trainer.step(self.opts.batch_size)

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                with autograd.record():
                    fake = self.g(latent_z)
                    output = self.d(fake).reshape((-1, 1))
                    err_g = loss(output, real_label)
                    err_g.backward()

                g_trainer.step(self.opts.batch_size)

                # Print log info every 10 batches
                if i % 10 == 0:
                    name, acc = metric.get()
                    logging.info('speed: {} samples/s'.format(self.opts.batch_size / (time.time() - btic)))
                    logging.info(
                        'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                        % (nd.mean(err_d).asscalar(),
                           nd.mean(err_g).asscalar(), acc, i, epoch))
                btic = time.time()

            name, acc = metric.get()
            metric.reset()

            logging.info('\nbinary training acc at epoch {}: {}={}'.format(epoch, name, acc))
            logging.info('time: {}'.format(time.time() - tic))

            # Visualize one generated image each 10 epoch
            if (epoch + 1) % 10 == 0:
                fake_img = fake[0]
                self.visualize(fake_img, epoch + 1)

            # Save models each 100 epochs
            if (epoch + 1) % 100 == 0:
                self.save_models(epoch)
        self.save_models(self.opts.epochs)

    def save_models(self, epoch):
        base_path = os.path.join(self.opts.output_dir, self.stamp, 'models')

        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        self.g.collect_params().save(os.path.join(base_path, 'd-{}-epochs.params'.format(epoch)))
        self.d.collect_params().save(os.path.join(base_path, 'g-{}-epochs.params'.format(epoch)))
