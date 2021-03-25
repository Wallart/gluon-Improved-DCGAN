from model import get_ctx, get_model
from dataset import datasets

from tqdm import tqdm
from mxnet import nd, gluon
from renderer import Renderer
from model.dc_gan.generator import Generator

import os
import random
import argparse

from trainer.dc_gan_trainer import DCGANTrainer


def render(opts):
    renderer = Renderer(opts)

    g = get_model(opts, opts.ctx, Generator, model_path=opts.g_model, symbol_path=opts.g_symbol)

    n_ctx = len(opts.ctx)
    img_per_ctx = opts.n_images // n_ctx
    print(f'Generating {img_per_ctx} image(s) per ctx. {n_ctx} ctx found.')
    for _ in tqdm(range(0, img_per_ctx)):
        latent_z = nd.random_normal(0, 1, shape=(n_ctx, opts.latent_z_size, 1, 1))
        latent_z = gluon.utils.split_and_load(latent_z, ctx_list=opts.ctx, batch_axis=0)

        fakes = [g(z).squeeze() for z in latent_z]
        for fake in fakes:
            img_name = str(random.getrandbits(128))
            renderer.render(fake, img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved Deep Convolutional Generative Adversarial Networks')
    sub_parsers = parser.add_subparsers(dest='action')

    train_parser = sub_parsers.add_parser('train')
    train_parser.add_argument('dataset_type', choices=datasets.keys(), type=str, help='training dataset type')
    train_parser.add_argument('path', type=str, help='path of the train dataset')
    train_parser.add_argument('image_size', type=int, help='size of the dataset images')
    train_parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=32, help='batch size')
    train_parser.add_argument('-b1', '--beta1', dest='beta1', type=float, default=0.5, help='batch1 value')
    train_parser.add_argument('-b2', '--beta2', dest='beta2', type=float, default=0.999, help='batch2 value')
    train_parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=1000, help='learning epochs')
    train_parser.add_argument('-o', '--output', dest='outdir', type=str, help='model output directory')
    train_parser.add_argument('-z', '--z-size', dest='latent_z_size', type=int, default=100, help='latent_z size')
    train_parser.add_argument('-c', '--colors', dest='num_colors', type=int, choices=[1, 3], default=3, help='number of colors for generated images')
    train_parser.add_argument('-r', '--relu', dest='relu', action='store_true', help='use old relu layers instead of selu')
    train_parser.add_argument('-w', '--workers', dest='workers', type=int, default=0, help='number of workers to use')
    train_parser.add_argument('--clip-gradient', dest='clip_gradient', type=float, default=10.0, help='clip gradient by projecting onto the box [-x, x]')
    train_parser.add_argument('--weight-decay', dest='wd', type=int, default=0, help='weight decay')

    train_parser.add_argument('--d-lr', dest='d_lr', type=float, default=0.00005, help='discriminator learning rate')
    train_parser.add_argument('--d-model', dest='d_model', type=str, help='discriminator model')
    train_parser.add_argument('--d-symbol', dest='d_symbol', type=str, help='discriminator symbol file path')

    train_parser.add_argument('--g-lr', dest='g_lr', type=float, default=0.0002, help='generator learning rate')
    train_parser.add_argument('--g-model', dest='g_model', type=str, help='generator model')
    train_parser.add_argument('--g-symbol', dest='g_symbol', type=str, help='generator symbol file path')

    train_parser.add_argument('--no-hybridize', dest='no_hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')
    train_parser.add_argument('--gpus', dest='gpus', type=str, default='', help='gpus id to use, for example 0,1')

    train_parser.add_argument('--cond', dest='conditional', action='store_true', help='enables conditional GAN training.')
    train_parser.add_argument('--embed-size', type=int, default=64, help='For conditional GAN. Label embed dim')

    train_parser.add_argument('--ndf', type=int, default=128, help='size of feature maps to handle in discriminator')
    train_parser.add_argument('--ngf', type=int, default=128, help='size of feature maps to produce in generator, whatever images size is')
    train_parser.add_argument('--ndr', type=int, help='number of rounds for the discriminator')
    train_parser.add_argument('--ngr', type=int, help='number of rounds for the generator')

    train_parser.add_argument('--overwrite', action='store_true', help='overwrite model if output directory already exists')
    train_parser.add_argument('--profile', action='store_true', help='enable profiling')

    train_parser.add_argument('--log-interval', dest='log_interval', type=int, help='iterations log interval')
    train_parser.add_argument('--extra-interval', dest='extra_interval', type=int, default=30, help='extra thumbnails interval generation (epochs)')
    train_parser.add_argument('--chkpt-interval', dest='chkpt_interval', type=int, help='model checkpointing interval (epochs)')
    train_parser.add_argument('--thumb-interval', dest='thumb_interval', type=int, default=1, help='thumbnail interval generation (epochs)')
    train_parser.add_argument('--viz-interval', dest='viz_interval', type=int, help='model visualization interval (epochs)')

    renderer_parser = sub_parsers.add_parser('render')
    renderer_parser.add_argument('g_model', type=str, help='generator model')
    renderer_parser.add_argument('image_size', type=int, help='size of the images to render')
    renderer_parser.add_argument('n_images', type=int, help='number of images to generate')
    renderer_parser.add_argument('output_dir', type=str, help='images output directory')

    renderer_parser.add_argument('--g-symbol', dest='g_symbol', type=str, help='generator symbol file path')

    renderer_parser.add_argument('-z', '--z-size', dest='latent_z_size', type=int, default=100, help='latent_z size')
    renderer_parser.add_argument('-c', '--colors', dest='num_colors', type=int, default=3, help='number of colors for generated images')
    renderer_parser.add_argument('-r', '--relu', dest='relu', action='store_true', help='use old relu layers instead of selu')
    renderer_parser.add_argument('--ngf', dest='ngf', type=int, default=128, help='number of hidden nodes in the generator')

    renderer_parser.add_argument('--overwrite', action='store_true', help='overwrite model if output directory already exists')
    renderer_parser.add_argument('--gpus', dest='gpus', type=str, default='', help='gpus id to use, for example 0,1')
    renderer_parser.add_argument('--no-hybridize', dest='no_hybridize', action='store_true', help='disable mxnet hybridize network (debug purpose)')

    args = parser.parse_args()
    if not hasattr(args, 'batch_size'):
        args.batch_size = 1

    args.ctx = get_ctx(args)
    if args.action == 'train':
        dataset = datasets[args.dataset_type](args)
        if args.conditional:
            args.num_classes = len(dataset._dataset.synsets)

        trainer = DCGANTrainer(args)
        trainer.train(dataset.get())
    elif args.action == 'render':
        render(args)
