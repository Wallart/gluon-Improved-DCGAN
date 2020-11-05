from abc import abstractmethod
from mxnet import nd, gluon, metric
from mxboard import SummaryWriter
from trainer.trainer import Trainer
from utils.metrics import facc

import os
import mxnet as mx


class GANTrainer(Trainer):

    def __init__(self, opts):
        super(GANTrainer, self).__init__(opts)

        self._d_rounds = opts.ndr or 1
        self._g_rounds = opts.ngr or 1

        self._d = self._build_discriminator(opts)
        self._g = self._build_generator(opts)

        self._acc_metric = metric.CustomMetric(facc)
        self._g_loss_metric = metric.Loss('generator_loss')
        self._d_loss_metric = metric.Loss('discriminator_loss')

        self._d_trainer = gluon.Trainer(self._d.collect_params(), 'Adam', {
            'learning_rate': self._opts.d_lr,
            'wd': self._opts.wd,
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'clip_gradient': self._opts.clip_gradient
        })

        self._g_trainer = gluon.Trainer(self._g.collect_params(), 'Adam', {
            'learning_rate': self._opts.g_lr,
            'wd': self._opts.wd,
            'beta1': self._opts.beta1,
            'beta2': self._opts.beta2,
            'clip_gradient': self._opts.clip_gradient
        })

    @abstractmethod
    def _build_discriminator(self, opts):
        pass

    @abstractmethod
    def _build_generator(self, opts):
        pass

    @abstractmethod
    def _sample_z(self):
        pass

    def _do_checkpoint(self, cur_epoch):
        outfile = os.path.join(self._outchkpts, '{}_{}-{:04d}.chkpt'.format(self.model_name(), 'generator', cur_epoch))
        self._g.save_parameters(outfile)

        outfile = os.path.join(self._outchkpts, '{}_{}-{:04d}.chkpt'.format(self.model_name(), 'discriminator', cur_epoch))
        self._d.save_parameters(outfile)

    def _export_model(self, num_epoch):
        outfile = os.path.join(self._outdir, '{}_generator'.format(self.model_name()))
        self._g.export(outfile, epoch=num_epoch)

        outfile = os.path.join(self._outdir, '{}_discriminator'.format(self.model_name()))
        self._d.export(outfile, epoch=num_epoch)

    def _visualize_graphs(self, cur_epoch, cur_iter):
        if not self._opts.no_hybridize and cur_epoch == 0 and cur_iter == 1:
            for net, out_path in self._networks:
                with SummaryWriter(logdir=out_path, flush_secs=5, verbose=False) as writer:
                    writer.add_graph(net)

    def _visualize_weights(self, writer, cur_epoch, cur_iter):
        if not self._opts.no_hybridize and self._viz_interval > 0 and cur_iter == 0 and cur_epoch % self._viz_interval == 0:
            for net, _ in self._networks:
                # to visualize gradients each x epochs
                params = [p for p in net.collect_params().values() if type(p) == gluon.Parameter and p._grad]
                for p in params:
                    name = '{}/{}/{}'.format(net._name, '_'.join(p.name.split('_')[:-1]), p.name.split('_')[-1])
                    aggregated_grads = nd.concat(*[grad.as_in_context(mx.cpu()) for grad in p._grad], dim=0)
                    writer.add_histogram(tag=name, values=aggregated_grads, global_step=cur_epoch + 1, bins=1000)
