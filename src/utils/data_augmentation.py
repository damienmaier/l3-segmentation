import functools

import tensorflow as tf
import tensorflow_addons as tfa


def generate_random_rotation_function(max_angle):
    """
    Returns a function.

    The functions takes as input an image and a mask as tf tensors of shape
    (<dim1>, <dim2>, 1). It applies an identical random rotation on both the image and the mask and it returns
    the results.

    `max_angle` is the maximal angle of the random rotation.
    """
    def random_rotation_function(image, mask):
        angle = tf.random.uniform(shape=(), minval=-max_angle, maxval=max_angle, dtype=tf.float32)
        rotation_function = functools.partial(tfa.image.rotate, angles=angle, interpolation="bilinear",
                                              fill_mode="nearest")
        rotated_image = rotation_function(images=image)
        rotated_mask = rotation_function(images=mask)
        return rotated_image, rotated_mask

    return random_rotation_function
