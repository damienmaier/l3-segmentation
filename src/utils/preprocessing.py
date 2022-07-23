import numpy as np


def triple_channels(image):
    return np.stack((image,) * 3, axis=-1)


def single_channel(image):
    return image[..., np.newaxis]


def clip(image):
    return np.clip(image, -200, 200)
