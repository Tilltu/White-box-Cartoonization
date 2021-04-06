import numpy as np
from keras.layers import DepthwiseConv2D


def box_filter(x, r):
    ch = x.shape[-1]
    weight = 1 / ((2 * r + 1) ** 2)
    box_kernel = weight * np.ones((2 * r + 1, 2 * r + 1, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = DepthwiseConv2D(box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = x.shape
