from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import profiler
from mxboard import SummaryWriter
from trainer.trainer import Trainer
from model.dc_gan.generator import Generator
from model.dc_gan.discriminator import Discriminator
from utils.metrics import facc
from utils.tensor import tensor_to_image, tensor_to_viz

import mxnet as mx
import logging
import os


class DCGANTrainer(Trainer):

    def __init__(self, opts):
        super(DCGANTrainer, self).__init__(opts)

        self.d = self._build_discriminator(opts)
        self.g = self._build_generator(opts)

        # It's more robust to compute sigmoid on the loss than in the discriminator
        self._binary_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

        self._acc_metric = mx.metric.CustomMetric(facc)
        self._g_loss_metric = mx.metric.Loss('generator_loss')
        self._d_loss_metric = mx.metric.Loss('discriminator_loss')

        self._hybridize()

        self._initialize(self.g, opts.g_model)
        self._g_trainer = gluon.Trainer(self.g.collect_params(), 'Adam', {
            'learning_rate': self._opts.g_lr,
            'wd': self._opts.wd,
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'clip_gradient': self._opts.clip_gradient
        })

        self._initialize(self.d, opts.d_model)
        self._d_trainer = gluon.Trainer(self.d.collect_params(), 'Adam', {
            'learning_rate': self._opts.d_lr,
            'wd': self._opts.wd,
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'clip_gradient': self._opts.clip_gradient
        })

    def model_name(self):
        return 'DCGAN'

    def _build_generator(self, opts):
        g = Generator(opts)
        logs = os.path.join(self._outlogs, 'generator')
        self._networks.append((g, logs))
        return g

    def _build_discriminator(self, opts):
        d = Discriminator(opts)
        logs = os.path.join(self._outlogs, 'discriminator')
        self._networks.append((d, logs))
        return d

    def train(self, train_data):
        fixed_noise = self.make_noise()

        real_label = gluon.utils.split_and_load(nd.ones((self._batch_size,)), ctx_list=self._opts.ctx, batch_axis=0)
        fake_label = gluon.utils.split_and_load(nd.zeros((self._batch_size,)), ctx_list=self._opts.ctx, batch_axis=0)

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
                    data = gluon.utils.split_and_load(d, ctx_list=self._opts.ctx, batch_axis=0)
                    # label = gluon.utils.split_and_load(l, ctx_list=self._opts.ctx, batch_axis=0)
                    noise = self.make_noise()
                    nd.waitall()

                    with autograd.record():
                        # Train with real image then fake image
                        r_output = [self.d(x) for x in data]
                        loss_d_real = [self.binary_loss(x, y) for x, y in zip(r_output, real_label)]
                        self._acc_metric.update(real_label, r_output)

                        f_output = [self.d(self.g(z).detach()) for z in noise]
                        loss_d_fake = [self.binary_loss(x, y) for x, y in zip(f_output, fake_label)]
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
                        loss_g = [self.binary_loss(self.d(f), y) for f, y in zip(fakes, real_label)]
                        self._g_loss_metric.update(0, loss_g)
                        autograd.backward(loss_g)

                    self._g_trainer.step(self._batch_size)

                    # Visualize generated image each x epoch over each gpus
                    if i == len(train_data) - 1 and (epoch + 1) % self._opts.thumb_interval == 0:
                        tensor_to_viz(writer, nd.concat(*fakes, dim=0), epoch + 1, 'Current_epoch_random_noise')

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
                if global_step % self._opts.extra_interval == 0:
                    tensor = nd.concat(*[self.g(z) for z in fixed_noise], dim=0)
                    tensor_to_viz(writer, tensor, global_step, 'Epoch_{}_fixed_noise'.format(global_step))
                    tensor_to_image(self._outimages, tensor, global_step)

                # save model each x epochs
                if global_step % self._chkpt_interval == 0:
                    self._do_checkpoint(global_step)

            nd.waitall()
            self._save_profile()
            self._export_model(self._epochs)

    def _initialize(self, net, pretrained=None):
        if pretrained:
            net.load_parameters(pretrained, ctx=self._opts.ctx)
        else:
            net.initialize(ctx=self._opts.ctx)

    def _hybridize(self):
        if not self._opts.no_hybridize:
            self.g.hybridize()
            self.d.hybridize()

    def make_noise(self):
        latent_z = nd.random.normal(0, 1, shape=(self._batch_size, self._opts.latent_z_size, 1, 1))
        return gluon.utils.split_and_load(latent_z, ctx_list=self._opts.ctx, batch_axis=0)

    def _export_model(self, num_epoch):
        outfile = os.path.join(self._outdir, '{}_generator'.format(self.model_name()))
        self.g.export(outfile, epoch=num_epoch)

        outfile = os.path.join(self._outdir, '{}_discriminator'.format(self.model_name()))
        self.d.export(outfile, epoch=num_epoch)

    def _do_checkpoint(self, cur_epoch):
        outfile = os.path.join(self._outchkpts, '{}_{}-{:04d}.chkpt'.format(self.model_name(), 'generator', cur_epoch))
        self.g.save_parameters(outfile)

        outfile = os.path.join(self._outchkpts, '{}_{}-{:04d}.chkpt'.format(self.model_name(), 'discriminator', cur_epoch))
        self.d.save_parameters(outfile)

    def _visualize_graphs(self, cur_epoch, cur_iter):
        if cur_epoch == 0 and cur_iter == 1:
            for net, out_path in self._networks:
                with SummaryWriter(logdir=out_path, flush_secs=5, verbose=False) as writer:
                    writer.add_graph(net)

    def _visualize_weights(self, writer, cur_epoch, cur_iter):
        if self._viz_interval > 0 and cur_iter == 0 and cur_epoch % self._viz_interval == 0:
            for net, _ in self._networks:
                # to visualize gradients each x epochs
                params = [p for p in net.collect_params().values() if type(p) == gluon.Parameter and p._grad]
                for p in params:
                    name = '{}/{}/{}'.format(net._name, '_'.join(p.name.split('_')[:-1]), p.name.split('_')[-1])
                    aggregated_grads = nd.concat(*[grad.as_in_context(mx.cpu()) for grad in p._grad], dim=0)
                    writer.add_histogram(tag=name, values=aggregated_grads, global_step=cur_epoch + 1, bins=1000)