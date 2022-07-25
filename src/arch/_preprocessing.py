import numpy as np


def triple_channels(image):
    """Transforms a numpy array of shape (length, width) to shape (length, width, 3)

    Each channel is a copy of the original array.
    """
    return np.stack((image,) * 3, axis=-1)


def single_channel(image):
    return image[..., np.newaxis]


def clip(image):
    return np.clip(image, -200, 200)