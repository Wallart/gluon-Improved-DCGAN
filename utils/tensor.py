from PIL import Image
from mxnet import nd

import os
import math
import mxnet as mx
import numpy as np


def agg_dist_tensor(tensor, ctx=mx.cpu()):
    if tensor is None:
        return None
    elif type(tensor) == list:
        return nd.concat(*[t.as_in_context(ctx) for t in tensor], dim=0)

    return tensor.as_in_context(ctx)


def tensor_to_viz(writer, tensor, epoch, tag):
    tensor = agg_dist_tensor(tensor)
    tensor = ((tensor.asnumpy() + 1.0) * 127.5).astype(np.uint8)
    writer.add_image(image=tensor, global_step=epoch, tag=tag)


def tensor_to_image(outdir, tensor, epoch):
    tensor = agg_dist_tensor(tensor)
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

    im = Image.fromarray(output)
    im.save(os.path.join(outdir, 'epoch-{:04d}.png'.format(epoch)))
