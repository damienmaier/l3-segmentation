import functools

import tensorflow as tf
import tensorflow_addons as tfa


def generate_random_rotation_function(max_angle):
    def random_rotation_function(image, mask):
        angle = tf.random.uniform(shape=(), minval=-max_angle, maxval=max_angle, dtype=tf.float32)
        rotation_function = functools.partial(tfa.image.rotate, angles=angle, interpolation="bilinear",
                                              fill_mode="nearest")
        rotated_image = rotation_function(images=image)
        rotated_mask = rotation_function(images=mask)
        return rotated_image, rotated_mask

    return random_rotation_function
