from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import profiler
from mxboard import SummaryWriter
from model.dc_gan.discriminator import Discriminator
from model.dc_gan.generator import Generator
from trainer.trainer import Trainer
from PIL import Image

import mxnet as mx
import numpy as np
import logging
import os
import math


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class DCGANTrainer(Trainer):

    def __init__(self, opts):
        super(DCGANTrainer, self).__init__(opts, 'DCGAN')

        self.opts = opts

        self.d = Discriminator(opts)
        self.g = Generator(opts)

        self.networks = [(self.g, self._outlogs_generator), (self.d, self._outlogs_discriminator)]

        # It's more robust to compute sigmoid on the loss than in the discriminator
        self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._acc_metric = mx.metric.CustomMetric(facc)
        self._g_loss_metric = mx.metric.Loss('generator_loss')
        self._d_loss_metric = mx.metric.Loss('discriminator_loss')

        self._hybridize()
        self._initialize(pretrained_g=opts.g_model, pretrained_d=opts.d_model)
        self._g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', {
            'learning_rate': self.opts.g_lr,
            'wd': self.opts.wd,
            'beta1': self.opts.beta1,
            'beta2': self.opts.beta2,
            'clip_gradient': self.opts.clip_gradient
        })
        self._d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', {
            'learning_rate': self.opts.d_lr,
            'wd': self.opts.wd,
            'beta1': self.opts.beta1,
            'beta2': self.opts.beta2,
            'clip_gradient': self.opts.clip_gradient
        })

    def train(self, train_data):
        fixed_noise = self.make_noise()

        real_label = gluon.utils.split_and_load(nd.ones((self._batch_size,)), ctx_list=self.opts.ctx, batch_axis=0)
        fake_label = gluon.utils.split_and_load(nd.zeros((self._batch_size,)), ctx_list=self.opts.ctx, batch_axis=0)

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
                    self._visualize_weights(writer, epoch)

                    if self._profile and epoch == 0 and i == 1:
                        profiler.set_state('run')

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    data = gluon.utils.split_and_load(d, ctx_list=self.opts.ctx, batch_axis=0)
                    # label = gluon.utils.split_and_load(l, ctx_list=self.opts.ctx, batch_axis=0)
                    noise = self.make_noise()
                    nd.waitall()

                    with autograd.record():
                        # Train with real image then fake image
                        r_output = [self.d(x) for x in data]
                        loss_d_real = [self.loss(x, y) for x, y in zip(r_output, real_label)]
                        self._acc_metric.update(real_label, r_output)

                        f_output = [self.d(self.g(z).detach()) for z in noise]
                        loss_d_fake = [self.loss(x, y) for x, y in zip(f_output, fake_label)]
                        self._acc_metric.update(fake_label, f_output)

                        loss_d = [r + f for r, f in zip(loss_d_real, loss_d_fake)]
                        self._d_loss_metric.update(0, loss_d)
                        autograd.backward(loss_d)

                    self._d_trainer.step(self._batch_size)

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################

                    with autograd.record():
                        fakes = [self.g(z) for z in noise]
                        loss_g = [self.loss(self.d(f), y) for f, y in zip(fakes, real_label)]
                        self._g_loss_metric.update(0, loss_g)
                        autograd.backward(loss_g)

                    self._g_trainer.step(self._batch_size)

                    # Visualize generated image each x epoch over each gpus
                    if i == len(train_data) - 1 and (epoch + 1) % self.opts.thumb_interval == 0:
                        self.tensor_to_viz(writer, nd.concat(*fakes, dim=0), epoch + 1, 'Current_epoch_random_noise')

                    # per x iter logging
                    if i % self._log_interval == 0:
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
                if global_step % self.opts.extra_interval == 0:
                    tensor = nd.concat(*[self.g(z) for z in fixed_noise], dim=0)
                    self.tensor_to_viz(writer, tensor, global_step, 'Epoch_{}_fixed_noise'.format(global_step))
                    self.tensor_to_image(tensor, global_step)

                # save model each x epochs
                if global_step % self._chkpt_interval == 0:
                    self._do_checkpoint(global_step)

            nd.waitall()
            self._save_profile()
            self._export_model(self._epochs)

    def tensor_to_viz(self, writer, tensor, epoch, tag):
        tensor = ((tensor.asnumpy() + 1.0) * 127.5).astype(np.uint8)
        writer.add_image(image=tensor, global_step=epoch, tag=tag)

    def tensor_to_image(self, tensor, epoch):
        images = tensor.asnumpy().transpose(0, 2, 3, 1)

        row = int(math.sqrt(len(images)))
        col = row
        height = sum(image.shape[0] for image in images[0:row])
        width = sum(image.shape[1] for image in images[0:col])
        output = np.zeros((height, width, 3))

        for i in range(row):
            for j in range(col):
                image = images[i * row + j]
                h, w, d = image.shape
                output[i * h:i * h + h, j * w:j * w + w] = image
        output = ((output + 1.0) * 127.5).astype(np.uint8)

        if not os.path.isdir(self._outimages):
            os.makedirs(self._outimages)

        im = Image.fromarray(output)
        im.save(os.path.join(self._outimages, 'epoch-{}.png'.format(epoch)))

    def _initialize(self, pretrained_g=None, pretrained_d=None):
        if pretrained_g:
            pretrained_g = os.path.expanduser(self.opts.g_model)
            self.g.load_parameters(pretrained_g, ctx=self.opts.ctx)
        else:
            self.g.initialize(ctx=self.opts.ctx)

        if pretrained_d:
            pretrained_d = os.path.expanduser(self.opts.d_model)
            self.d.load_parameters(pretrained_d, ctx=self.opts.ctx)
        else:
            self.d.initialize(ctx=self.opts.ctx)

    def _hybridize(self):
        if not self.opts.no_hybridize:
            self.g.hybridize()
            self.d.hybridize()

    def make_noise(self):
        latent_z = nd.random.normal(0, 1, shape=(self._batch_size, self.opts.latent_z_size, 1, 1))
        return gluon.utils.split_and_load(latent_z, ctx_list=self.opts.ctx, batch_axis=0)

    def _export_model(self, num_epoch):
        outfile = os.path.join(self._outdir, 'DCGAN_generator')
        self.g.export(outfile, epoch=num_epoch)

        outfile = os.path.join(self._outdir, 'DCGAN_discriminator')
        self.d.export(outfile, epoch=num_epoch)

    def _do_checkpoint(self, cur_epoch):
        outfile = os.path.join(self._outchkpts, '{}-{:04d}.chkpt'.format('DCGAN_generator', cur_epoch))
        self.g.save_parameters(outfile)

        outfile = os.path.join(self._outchkpts, '{}-{:04d}.chkpt'.format('DCGAN_discriminator', cur_epoch))
        self.d.save_parameters(outfile)

    def _visualize_graphs(self, cur_epoch, cur_iter):
        if cur_epoch == 0 and cur_iter == 1:
            for net, out_path in self.networks:
                with SummaryWriter(logdir=out_path, flush_secs=5, verbose=False) as writer:
                    writer.add_graph(net)

    def _visualize_weights(self, writer, cur_epoch):
        if self._viz_interval > 0 and cur_epoch % self._viz_interval == 0:
            for net, _ in self.networks:
                # to visualize gradients each x epochs
                params = [p for p in net.collect_params().values() if type(p) == gluon.Parameter and p._grad]
                for p in params:
                    name = '{}/{}/{}'.format(net._name, '_'.join(p.name.split('_')[:-1]), p.name.split('_')[-1])
                    aggregated_grads = nd.concat(*[grad.as_in_context(mx.cpu()) for grad in p._grad], dim=0)
                    writer.add_histogram(tag=name, values=aggregated_grads, global_step=cur_epoch + 1, bins=1000)