import mxnet as mx
import os


class Options:

    def __init__(self, image_size, data_root, output_dir, use_gpu=True):
        try:
            self.ctx = mx.gpu() if use_gpu else mx.cpu()
        except:
            self.ctx = mx.cpu()

        self.data_root = data_root
        self.output_dir = os.path.join(output_dir)
        self.epochs = 400
        # Higher batch size decreases generalisation capabilities
        self.batch_size = 32
        self.img_size = image_size
        self.latent_z_size = 100
        self.num_colors = 3
        self.with_selu = True
        self.g_lr = 0.0002
        self.d_lr = 0.00005
        self.g_h_size = 64
        self.d_h_size = 64
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.wd = 0
        self.clip_gradient = 10.0
        self.visualize = True
        self.hybridize = True
        self.checkpoint_interval = 25
        self.graph_to_display = 'generator'
        self.thumb_interval = 1
        self.extra_img_interval = 30
