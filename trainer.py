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
import shutil
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
        # from_sigmoid is required, or we have to remove the sigmoid activation layer in the discriminator network
        self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.metric = mx.metric.CustomMetric(facc)

        self.sw = None
        self.stamp = 'DC-GAN-{}e-{}'.format(self.opts.epochs, datetime.now().strftime('%y_%m_%d-%H_%M'))
        if self.opts.model_name:
            self.stamp = self.opts.model_name

        output_dir = os.path.expanduser(self.opts.output_dir)
        output_dir = output_dir if os.path.abspath(output_dir) else os.path.join(os.getcwd(), self.opts.output_dir)

        path = os.path.join(output_dir, self.stamp)
        if os.path.isdir(path) and self.opts.overwrite:
            shutil.rmtree(path)
            os.makedirs(path)
        elif os.path.isdir(path):
            raise Exception('Output directory already exists.')

        self.logs_path = os.path.join(path, 'logs')
        self.images_path = os.path.join(path, 'images')
        self.models_path = os.path.join(path, 'models')

        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    def train(self, train_data):
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

        latent_z = nd.random_normal(0, 1, shape=(self.opts.batch_size, self.opts.latent_z_size, 1, 1))
        latent_z = gluon.utils.split_and_load(latent_z, ctx_list=self.opts.ctx, batch_axis=0)
        real_label = gluon.utils.split_and_load(nd.ones((self.opts.batch_size,)), ctx_list=self.opts.ctx, batch_axis=0)
        fake_label = gluon.utils.split_and_load(nd.zeros((self.opts.batch_size,)), ctx_list=self.opts.ctx, batch_axis=0)

        with SummaryWriter(logdir=self.logs_path, flush_secs=5, verbose=False) as self.sw:
            for epoch in range(self.opts.epochs):
                if not self.opts.no_hybridize and epoch == 1:
                    if self.opts.graph == 'generator':
                        self.sw.add_graph(self.g)
                    elif self.opts.graph == 'discriminator':
                        self.sw.add_graph(self.d)

                e_tic = time.time()
                for i, (d, l) in enumerate(train_data):
                    b_tic = time.time()

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    data = gluon.utils.split_and_load(d, ctx_list=self.opts.ctx, batch_axis=0)
                    # label = gluon.utils.split_and_load(l, ctx_list=self.opts.ctx, batch_axis=0)
                    nd.waitall()

                    with autograd.record():
                        # Train with real image then fake image
                        r_output = [self.d(x) for x in data]
                        loss_d_real = [self.loss(x, y) for x, y in zip(r_output, real_label)]
                        self.metric.update(real_label, r_output)

                        f_output = [self.d(self.g(z).detach()) for z in latent_z]
                        loss_d_fake = [self.loss(x, y) for x, y in zip(f_output, fake_label)]
                        self.metric.update(fake_label, f_output)

                        loss_d = [r + f for r, f in zip(loss_d_real, loss_d_fake)]
                        for loss in loss_d:
                            loss.backward()

                    d_trainer.step(self.opts.batch_size)

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################

                    with autograd.record():
                        fakes = [self.g(z) for z in latent_z]
                        loss_g = [self.loss(self.d(f), y) for f, y in zip(fakes, real_label)]
                        for loss in loss_g:
                            loss.backward()

                    g_trainer.step(self.opts.batch_size)

                    loss_d_scalar = sum([e.mean().asscalar() for e in loss_d]) / len(loss_d)
                    loss_g_scalar = sum([e.mean().asscalar() for e in loss_g]) / len(loss_g)

                    # Visualize generated image each x epoch over each gpus
                    if (epoch + 1) % self.opts.thumb_interval == 0:
                        for fake in fakes:
                            self.generate_thumb('epoch_thumbnail', fake[0], epoch + 1)

                    # Print log info every 10 batches
                    if i % 10 == 0:
                        name, acc = self.metric.get()
                        batch_time = time.time() - b_tic
                        logging.info('[Epoch {}][Iter {}]'.format(epoch + 1, i, ))
                        logging.info('\tD_loss = {:.6f}, G_loss = {:.6f}, Acc = {:.6f}'.format(loss_d_scalar, loss_g_scalar, acc))
                        logging.info('\tTime: {:.2f} second(s)'.format(batch_time))
                        logging.info('\tSpeed: {:.2f} samples/s'.format(self.opts.batch_size / batch_time))

                    # Do epoch logging
                    if i == len(train_data) - 1:
                        _, acc = self.metric.get()
                        self.metric.reset()
                        logging.info('\n[Epoch {}] Acc = {:.6f} Time: {:.2f}\n'.format(epoch + 1, acc, time.time() - e_tic))
                        self.sw.add_scalar(tag='generator_loss', value=loss_g_scalar, global_step=epoch)
                        self.sw.add_scalar(tag='discriminator_loss', value=loss_d_scalar, global_step=epoch)
                        self.sw.add_scalar(tag='accuracy', value=acc, global_step=epoch)

                # TODO Fix extra thumbs
                # Generate batch_size random images each x epochs
                #if (epoch + 1) % self.opts.extra_interval == 0:
                #    self.extra_thumb(epoch + 1)

                # Save models each x epochs
                if (epoch + 1) % self.opts.checkpoint_interval == 0:
                    self.save_models(epoch + 1)

            nd.waitall()
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
            self.g.initialize(mx.init.Normal(0.02), ctx=self.opts.ctx)

        if self.opts.d_model:
            d_model = os.path.expanduser(self.opts.d_model)
            self.d.load_parameters(d_model, ctx=self.opts.ctx)
        else:
            self.d.initialize(mx.init.Normal(0.02), ctx=self.opts.ctx)

    def hybridize(self):
        if not self.opts.no_hybridize:
            self.g.hybridize()
            self.d.hybridize()
