from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxboard import SummaryWriter
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

        self.sw = None
        self.stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

        self.logs_path = os.path.join(self.opts.output_dir, self.stamp, 'logs')
        self.images_path = os.path.join(self.opts.output_dir, self.stamp, 'images')

        logging.basicConfig(level=logging.DEBUG)

    def visualize(self, tag, img_arr, epoch):
        if self.opts.visualize:
            image = ((img_arr + 1.0) * 127.5).astype(np.uint8)
            self.sw.add_image(tag=tag, image=image, global_step=epoch)
        else:
            plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
            plt.axis('off')

            if not os.path.isdir(self.images_path):
                os.makedirs(self.images_path)

            plt.savefig(os.path.join(self.images_path, '{}-{}.png'.format(tag, epoch)))

    def train(self, train_data):

        real_label = nd.ones((self.opts.batch_size,), ctx=self.opts.ctx)
        fake_label = nd.zeros((self.opts.batch_size,), ctx=self.opts.ctx)

        metric = mx.metric.CustomMetric(facc)

        self.hybridize()
        self.g.initialize(ctx=self.opts.ctx)
        self.d.initialize(ctx=self.opts.ctx)

        g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', {
            'learning_rate': self.opts.g_lr,
            'wd': self.opts.wd,
            'beta1': 0.5,
            'beta2': 0.999,
            'clip_gradient': self.opts.clip_gradient
        })
        d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', {
            'learning_rate': self.opts.d_lr,
            'wd': self.opts.wd,
            'beta1': 0.5,
            'beta2': 0.999,
            'clip_gradient': self.opts.clip_gradient
        })

        loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        latent_z = nd.random_normal(0, 1, shape=(self.opts.batch_size, self.opts.latent_z_size, 1, 1), ctx=self.opts.ctx)

        with SummaryWriter(logdir=self.logs_path, flush_secs=5, verbose=False) as self.sw:
            for epoch in range(self.opts.epochs):
                fake = None
                tic = time.time()
                btic = time.time()

                d_loss_scalar = 0
                g_loss_scalar = 0

                if self.opts.hybridize and epoch == 1:
                    if self.opts.graph_to_display == 'generator':
                        self.sw.add_graph(self.g)
                    elif self.opts.graph_to_display == 'discriminator':
                        self.sw.add_graph(self.d)

                for i, (d, l) in enumerate(train_data):
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    data = d.as_in_context(self.opts.ctx)

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

                    d_loss_scalar = nd.mean(err_d).asscalar()
                    g_loss_scalar = nd.mean(err_g).asscalar()

                    # Print log info every 10 batches
                    if i % 10 == 0:
                        name, acc = metric.get()
                        logging.info('speed: {} samples/s'.format(self.opts.batch_size / (time.time() - btic)))
                        logging.info(
                            'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                            % (d_loss_scalar, g_loss_scalar, acc, i, epoch))

                    btic = time.time()

                name, acc = metric.get()
                metric.reset()

                # Log to tensorboard
                self.sw.add_scalar(tag='generator_loss', value=g_loss_scalar, global_step=epoch)
                self.sw.add_scalar(tag='discriminator_loss', value=d_loss_scalar, global_step=epoch)
                self.sw.add_scalar(tag='accuracy', value=acc, global_step=epoch)

                logging.info('\nbinary training acc at epoch {}: {}={}'.format(epoch, name, acc))
                logging.info('time: {}'.format(time.time() - tic))

                # Visualize one generated image each x epoch
                if (epoch + 1) % self.opts.thumb_interval == 0:
                    fake_img = fake[0]
                    self.visualize('Training_thumbnail', fake_img, epoch + 1)

                # Generate batch_size random images each x epochs
                if (epoch + 1) % self.opts.extra_img_interval == 0:
                    self.random_extra_images(epoch + 1)

                # Save models each x epochs
                if (epoch + 1) % self.opts.checkpoint_interval == 0:
                    self.save_models(epoch + 1)
            self.save_models(self.opts.epochs)

    def save_models(self, epoch):
        base_path = os.path.join(self.opts.output_dir, self.stamp, 'models')

        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        self.d.collect_params().save(os.path.join(base_path, 'g-{}-epochs.params'.format(epoch)))
        self.g.collect_params().save(os.path.join(base_path, 'd-{}-epochs.params'.format(epoch)))

    def random_extra_images(self, epoch):
        for i in range(0, self.opts.batch_size):
            latent_z = nd.random_normal(0, 1, shape=(self.opts.batch_size, self.opts.latent_z_size, 1, 1), ctx=self.opts.ctx)
            fake = self.g(latent_z)
            self.visualize('Epoch_{}'.format(epoch), fake[0], epoch)


    def hybridize(self):
        if self.opts.hybridize:
            self.g.hybridize()
            self.d.hybridize()
