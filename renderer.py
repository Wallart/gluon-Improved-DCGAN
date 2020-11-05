from PIL import Image

import os
import shutil
import numpy as np


class Renderer:

    def __init__(self, opts):
        self._output_dir = os.path.expanduser(opts.output_dir)

        if opts.overwrite:
            shutil.rmtree(self._output_dir, ignore_errors=True)
        os.makedirs(self._output_dir)

    def render(self, img_arr, img_name):
        img_arr = ((img_arr.transpose((1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
        im = Image.fromarray(img_arr.asnumpy())
        im.save(os.path.join(self._output_dir, f'{img_name}.jpg'))
