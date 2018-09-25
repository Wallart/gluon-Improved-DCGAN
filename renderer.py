import numpy as np
import matplotlib.pyplot as plt
import os

class Renderer:

    @staticmethod
    def render(img_arr, img_name, output_dir):
        plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.axis('off')

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir, img_name))
