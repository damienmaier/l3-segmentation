import keras.layers
import tensorflow as tf


class ClipLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return tf.clip_by_value(inputs, -200, 200)


class RoundLayer(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.math.round(inputs)


class GrayscaleToRGBLayer(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.image.grayscale_to_rgb(inputs)
