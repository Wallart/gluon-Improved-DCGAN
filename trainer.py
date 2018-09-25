from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxboard import SummaryWriter
from datetime import datetime
from discriminator import Discriminator
from generator import Generator
from renderer import Renderer

import mxnet as mx
import numpy as np
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
        self.stamp = 'DC-GAN-{}e-{}'.format(self.opts.epochs, datetime.now().strftime('%y_%m_%d-%H_%M'))

        output_dir = os.path.expanduser(self.opts.output_dir)
        output_dir = output_dir if os.path.abspath(output_dir) else os.path.join(os.getcwd(), self.opts.output_dir)

        self.logs_path = os.path.join(output_dir, self.stamp, 'logs')
        self.images_path = os.path.join(output_dir, self.stamp, 'images')
        self.models_path = os.path.join(output_dir, self.stamp, 'models')

        logging.basicConfig(level=logging.DEBUG)

    def train(self, train_data):
        real_label = nd.ones((self.opts.batch_size,), ctx=self.opts.ctx)
        fake_label = nd.zeros((self.opts.batch_size,), ctx=self.opts.ctx)

        metric = mx.metric.CustomMetric(facc)

        self.hybridize()
        self.initialize()

        g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', {
            'learning_rate': self.opts.g_lr,
            'wd': self.opts.wd,
            'beta1': self.opts.beta1,
            'beta2': self.opts.beta2,
            'clip_gradient': self.opts.clip_gradient
        })
        d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', {
            'learning_rate': self.opts.d_lr,
            'wd': self.opts.wd,
            'beta1': self.opts.beta1,
            'beta2': self.opts.beta2,
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

                if not self.opts.no_hybridize and epoch == 1:
                    if self.opts.graph == 'generator':
                        self.sw.add_graph(self.g)
                    elif self.opts.graph == 'discriminator':
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
                    self.generate_thumb('epoch_thumbnail', fake_img, epoch + 1)

                # Generate batch_size random images each x epochs
                if (epoch + 1) % self.opts.extra_interval == 0:
                    self.extra_thumb(epoch + 1)

                # Save models each x epochs
                if (epoch + 1) % self.opts.checkpoint_interval == 0:
                    self.save_models(epoch + 1)
            self.save_models(self.opts.epochs)

    def save_models(self, epoch):
        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        self.d.save_parameters(os.path.join(self.models_path, 'd-{}-epochs.params'.format(epoch)))
        self.g.save_parameters(os.path.join(self.models_path, 'g-{}-epochs.params'.format(epoch)))

    def generate_thumb(self, tag, img_arr, epoch):
        if not self.opts.no_visualize:
            image = ((img_arr + 1.0) * 127.5).astype(np.uint8)
            self.sw.add_image(tag=tag, image=image, global_step=epoch)
        else:
            img_name = '{}-{}.png'.format(tag, epoch)
            Renderer.render(img_arr, img_name, self.images_path)

    def extra_thumb(self, epoch):
        for i in range(0, self.opts.batch_size):
            latent_z = nd.random_normal(0, 1, shape=(self.opts.batch_size, self.opts.latent_z_size, 1, 1), ctx=self.opts.ctx)
            fake = self.g(latent_z)
            self.generate_thumb('Epoch_{}'.format(epoch), fake[0], epoch)

    def initialize(self):
        if self.opts.g_model:
            g_model = os.path.expanduser(self.opts.g_model)
            self.g.load_parameters(g_model, ctx=self.opts.ctx)
        else:
            self.g.initialize(ctx=self.opts.ctx)

        if self.opts.d_model:
            d_model = os.path.expanduser(self.opts.d_model)
            self.d.load_parameters(d_model, ctx=self.opts.ctx)
        else:
            self.d.initialize(ctx=self.opts.ctx)

    def hybridize(self):
        if not self.opts.no_hybridize:
            self.g.hybridize()
            self.d.hybridize()
