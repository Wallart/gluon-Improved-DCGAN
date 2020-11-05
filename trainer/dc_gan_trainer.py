from model import get_model
from mxboard import SummaryWriter
from trainer.gan_trainer import GANTrainer
from model.dc_gan.generator import Generator
from mxnet import nd, gluon, profiler, autograd
from model.dc_gan.discriminator import Discriminator
from utils.tensor import tensor_to_image, tensor_to_viz

import os
import logging


class DCGANTrainer(GANTrainer):

    def __init__(self, opts):
        super(DCGANTrainer, self).__init__(opts)

        # It's more robust to compute sigmoid on the loss than in the discriminator
        self._binary_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

    def _build_discriminator(self, opts):
        d = get_model(opts, opts.ctx, Discriminator, model_path=opts.d_model, symbol_path=opts.d_symbol)
        logs = os.path.join(self._outlogs, 'discriminator')
        self._networks.append((d, logs))
        return d

    def _build_generator(self, opts):
        g = get_model(opts, opts.ctx, Generator, model_path=opts.g_model, symbol_path=opts.g_symbol)
        logs = os.path.join(self._outlogs, 'generator')
        self._networks.append((g, logs))
        return g

    def train(self, train_data):
        truth_labels = gluon.utils.split_and_load(nd.ones((self._batch_size,)), ctx_list=self._opts.ctx, batch_axis=0)
        fake_labels = gluon.utils.split_and_load(nd.zeros((self._batch_size,)), ctx_list=self._opts.ctx, batch_axis=0)
        fixed_z_samples = self._sample_z()

        with SummaryWriter(logdir=self._outlogs, flush_secs=5, verbose=False) as writer:
            num_iter = len(train_data)
            for epoch in range(self._epochs):
                self.e_tick()

                self._acc_metric.reset()
                self._g_loss_metric.reset()
                self._d_loss_metric.reset()

                for i, (d, l) in enumerate(train_data):
                    self.b_tick()

                    self._visualize_graphs(epoch, i)
                    self._visualize_weights(writer, epoch, i)

                    if self._profile and epoch == 0 and i == 1:
                        profiler.set_state('run')

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    images = gluon.utils.split_and_load(d, ctx_list=self._opts.ctx, batch_axis=0)
                    # label = gluon.utils.split_and_load(l, ctx_list=self._opts.ctx, batch_axis=0)
                    z_samples = self._sample_z()

                    with autograd.record():
                        # Train with real image then fake image
                        discriminated_images = [self._d(x) for x in images]
                        loss_d_truth = [self._binary_loss(x, y) for x, y in zip(discriminated_images, truth_labels)]
                        self._acc_metric.update(truth_labels, discriminated_images)

                        discriminated_fakes = [self._d(self._g(z).detach()) for z in z_samples]
                        loss_d_fake = [self._binary_loss(x, y) for x, y in zip(discriminated_fakes, fake_labels)]
                        self._acc_metric.update(fake_labels, discriminated_fakes)

                        loss_d = [t + f for t, f in zip(loss_d_truth, loss_d_fake)]
                        self._d_loss_metric.update(0, loss_d)

                    if (i + 1) % self._d_rounds == 0:
                        # Don't compute the mean of loss
                        # because SigmoidBinaryCrossEntropyLoss is already returning a mean
                        autograd.backward(loss_d)
                        self._d_trainer.step(self._batch_size)

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################

                    with autograd.record():
                        fakes = [self._g(z) for z in z_samples]
                        loss_g = [self._binary_loss(self._d(f), y) for f, y in zip(fakes, truth_labels)]
                        self._g_loss_metric.update(0, loss_g)

                    if (i + 1) % self._g_rounds == 0:
                        # Don't compute the mean of loss
                        # because SigmoidBinaryCrossEntropyLoss is already returning a mean
                        autograd.backward(loss_g)
                        self._g_trainer.step(self._batch_size)

                    # Visualize generated image each x epoch over each gpus
                    if (i + 1) == num_iter and (epoch + 1) % self._opts.thumb_interval == 0:
                        tensor_to_viz(writer, fakes, epoch + 1, 'current_z_samples')

                    # per x iter logging
                    if (i + 1) % self._log_interval == 0:
                        b_time = self.b_ellapsed()
                        speed = self._batch_size / b_time
                        iter_stats = 'exec time: {:.2f} second(s) speed: {:.2f} samples/s'.format(b_time, speed)
                        self._log('[Epoch {}] --[{}/{}]-- {}'.format(epoch + 1, i + 1, num_iter, iter_stats))

                # per epoch logging
                _, loss_g_scalar = self._g_loss_metric.get()
                _, loss_d_scalar = self._d_loss_metric.get()
                _, acc = self._acc_metric.get()

                global_step = epoch + 1
                self._log('[Epoch {}] exec time: {:.2f}'.format(global_step, self.e_ellapsed()))
                logging.info('\tD_loss = {:.6f}, G_loss = {:.6f}'.format(loss_d_scalar, loss_g_scalar))
                logging.info('\tacc = {:.6f}'.format(acc))

                # per epoch reporting
                losses = {'generator': loss_g_scalar, 'discriminator': loss_d_scalar}
                writer.add_scalar(tag='losses', value=losses, global_step=global_step)
                writer.add_scalar(tag='acc', value=acc, global_step=global_step)

                # generate batch_size random images each x epochs
                if global_step % self._opts.extra_interval == 0:
                    extra_fakes = [self._g(z) for z in fixed_z_samples]
                    tensor_to_viz(writer, extra_fakes, global_step, 'fixed_samples')
                    tensor_to_image(self._outimages, extra_fakes, global_step)

                # save model each x epochs
                if global_step % self._chkpt_interval == 0:
                    self._do_checkpoint(global_step)

            nd.waitall()
            self._save_profile()
            self._export_model(self._epochs)

    def model_name(self):
        return 'DCGAN'

    def _sample_z(self):
        latent_z = nd.random.normal(0, 1, shape=(self._batch_size, self._opts.latent_z_size, 1, 1))
        return gluon.utils.split_and_load(latent_z, ctx_list=self._opts.ctx, batch_axis=0)
