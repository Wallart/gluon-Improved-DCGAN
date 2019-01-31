from abc import ABC, abstractmethod
from datetime import datetime
from mxnet import profiler

import os
import time
import shutil
import logging


class TrainerException(Exception):
    pass


class Trainer(ABC):

    def __init__(self, opts, model_name):
        self._epochs = opts.epochs
        self._batch_size = opts.batch_size
        self._ctx = opts.ctx

        self._log_interval = opts.log_interval or 5
        self._chkpt_interval = opts.chkpt_interval or 30
        #self._viz_interval = opts.viz_interval or 2
        #self._profile = opts.profile

        self._epoch_tick = 0
        self._batch_tick = 0

        self._overwrite = opts.overwrite
        self._outdir = opts.outdir or os.path.join(os.getcwd(), '{}-{}e-{}'.format(model_name, self._epochs, datetime.now().strftime('%y_%m_%d-%H_%M')))
        self._outdir = os.path.expanduser(self._outdir)
        self._outlogs = os.path.join(self._outdir, 'logs')
        self._outchkpts = os.path.join(self._outdir, 'checkpoints')
        self._outimages = os.path.join(self._outdir, 'images')
        self._prepare_outdir()

        #if self._profile:
        #    self._outprofile = os.path.join(self._outdir, 'profile.json')
        #    profiler.set_config(profile_all=True, aggregate_stats=True, filename=self._outprofile)

        logging.basicConfig()
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

    def _prepare_outdir(self):
        outdir_exists = os.path.isdir(self._outdir)
        if outdir_exists and not self._overwrite:
            raise TrainerException('Output directory already exists.')
        elif os.path.isdir(self._outdir) and self._overwrite:
            shutil.rmtree(self._outdir)

        os.makedirs(self._outlogs)
        os.makedirs(self._outchkpts)
        os.makedirs(self._outimages)

    def b_tick(self):
        self._batch_tick = time.time()

    def e_tick(self):
        self._epoch_tick = time.time()

    def b_ellapsed(self):
        return time.time() - self._batch_tick

    def e_ellapsed(self):
        return time.time() - self._epoch_tick

    def _log(self, message, level=logging.INFO):
        self._logger.log(level, message)

    def _save_profile(self):
        if self._profile:
            print(profiler.dumps())
            profiler.dump()

    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def _export_model(self, num_epoch):
        pass

    @abstractmethod
    def _do_checkpoint(self, cur_epoch):
        pass

    @abstractmethod
    def _initialize(self, pretrained_g, pretrained_d):
        pass
