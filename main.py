from mxnet import gluon
from mxnet import nd
from generator import Generator
from renderer import Renderer
from trainer import Trainer
from tqdm import tqdm

import mxnet as mx
import numpy as np
import argparse
import logging
import random
import os


def get_transformer(opts):
    def transformer(data, label):
        data = mx.image.imresize(data, opts.image_size, opts.image_size)
        data = nd.transpose(data, (2, 0, 1))
        data = (data.astype(np.float32) / 128.0) - 1.0
        if data.shape[0] == 1:
            data = nd.tile(data, (3, 1, 1))
        return data, label
    return transformer


def get_dataset_from_folder(opts):
    func = get_transformer(opts)
    train_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(opts.dataset, transform=func),
                                       batch_size=opts.batch_size, shuffle=True,
                                       last_batch='discard', num_workers=opts.workers)
    test_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(opts.dataset, transform=func),
                                      batch_size=opts.batch_size, shuffle=False,
                                      last_batch='discard', num_workers=opts.workers)

    return train_data, test_data


def render(opts):
    model = os.path.expanduser(opts.generator)
    output_dir = os.path.expanduser(opts.out_dir)
    output_dir = output_dir if os.path.abspath(output_dir) else os.path.join(os.getcwd(), opts.out_dir)
    gen = Generator(opts)
    gen.load_parameters(model, ctx=opts.ctx)

    n_ctx = len(opts.ctx)
    for _ in tqdm(range(0, opts.n_images // n_ctx)):
        latent_z = nd.random_normal(0, 1, shape=(n_ctx, opts.latent_z_size, 1, 1))
        latent_z = gluon.utils.split_and_load(latent_z, ctx_list=opts.ctx, batch_axis=0)
        for z in latent_z:
            fakes = gen(z)
            for fake in fakes:
                img_name = str(random.getrandbits(128))
                Renderer.render(fake, img_name, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved Deep Convolutional Generative Adversarial Networks')
    sub_parsers = parser.add_subparsers(dest='action')

    train_parser = sub_parsers.add_parser('train')
    train_parser.add_argument('dataset', type=str, help='path of the train dataset')
    train_parser.add_argument('image_size', type=int, help='size of the dataset images')
    train_parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=32, help='batch size')
    train_parser.add_argument('-b1', '--beta1', dest='beta1', type=float, default=0.5, help='batch1 value')
    train_parser.add_argument('-b2', '--beta2', dest='beta2', type=float, default=0.999, help='batch2 value')
    train_parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=1000, help='learning epochs')
    train_parser.add_argument('-n', '--name', dest='model_name', type=str, help='model name')
    train_parser.add_argument('-o', '--output', dest='output_dir', type=str, default=os.getcwd(), help='model output directory')
    train_parser.add_argument('-z', '--z-size', dest='latent_z_size', type=int, default=100, help='latent_z size')
    train_parser.add_argument('-c', '--colors', dest='num_colors', type=int, default=3, help='number of colors for generated images')
    train_parser.add_argument('-r', '--relu', dest='relu', action='store_true', help='use old relu layers instead of selu')
    train_parser.add_argument('-w', '--workers', dest='workers', type=int, default=0, help='number of workers to use')
    train_parser.add_argument('--checkpoint-interval', dest='checkpoint_interval', type=int, default=25, help='models checkpointing interval (epochs)')
    train_parser.add_argument('--clip-gradient', dest='clip_gradient', type=float, default=10.0, help='clip gradient by projecting onto the box [-x, x]')
    train_parser.add_argument('--d-lr', dest='d_lr', type=float, default=0.00005, help='discriminator learning rate')
    train_parser.add_argument('--d-model', dest='d_model', type=str, help='discriminator model')
    train_parser.add_argument('--disable-hybridize', dest='no_hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')
    train_parser.add_argument('--disable-visualize', dest='no_visualize', action='store_true', help='disable images in mxboard. Save on disk')
    train_parser.add_argument('--extra-interval', dest='extra_interval', type=int, default=30, help='extra thumbnails interval generation (epochs)')
    train_parser.add_argument('--g-lr', dest='g_lr', type=float, default=0.0002, help='generator learning rate')
    train_parser.add_argument('--g-model', dest='g_model', type=str, help='generator model')
    train_parser.add_argument('--graph', dest='graph', type=str, choices=['discriminator', 'generator'], default='generator', help='network to render in mxboard')
    train_parser.add_argument('--gpus', dest='gpus', type=str, default='', help='gpus id to use, fot example 0,1')
    train_parser.add_argument('--ndf', type=int, default=128, help='size of feature maps to handle in discriminator')
    train_parser.add_argument('--ngf', type=int, default=128, help='size of feature maps to produce in generator, whatever images size is')
    train_parser.add_argument('--overwrite', action='store_true', help='overwrite model if output directory already exists')
    train_parser.add_argument('--thumb-interval', dest='thumb_interval', type=int, default=1, help='thumbnail interval generation (epochs)')
    train_parser.add_argument('--weight-decay', dest='wd', type=int, default=0, help='weight decay')

    renderer_parser = sub_parsers.add_parser('render')
    renderer_parser.add_argument('generator', type=str, help='generator model')
    renderer_parser.add_argument('image_size', type=int, help='size of the images to render')
    renderer_parser.add_argument('n_images', type=int, help='number of images to generate')
    renderer_parser.add_argument('-c', '--colors', dest='num_colors', type=int, default=3, help='number of colors for generated images')
    renderer_parser.add_argument('-o', '--output', dest='out_dir', type=str, default=os.getcwd(), help='images output directory')
    renderer_parser.add_argument('-r', '--relu', dest='relu', action='store_true', help='use old relu layers instead of selu')
    renderer_parser.add_argument('-z', '--z-size', dest='latent_z_size', type=int, default=100, help='latent_z size')
    renderer_parser.add_argument('--ngf', dest='ngf', type=int, default=128, help='number of hidden nodes in the generator')
    renderer_parser.add_argument('--gpus', dest='gpus', type=str, default='', help='gpus id to use, fot example 0,1')

    args = parser.parse_args()
    try:
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        args.ctx = args.ctx if args.ctx else [mx.cpu()]
    except:
        logging.error('Cannot access GPU, fallback to CPU')
        args.ctx = [mx.cpu()]

    if args.action == 'train':
        train_dataset, _ = get_dataset_from_folder(args)
        trainer = Trainer(args)
        trainer.train(train_dataset)
    elif args.action == 'render':
        render(args)
