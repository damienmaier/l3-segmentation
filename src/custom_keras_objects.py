"""
Provides custom keras layers and metrics.
"""

import keras.layers
import keras.utils
import tensorflow as tf

import model_evaluation


# The @tf.keras.utils.register_keras_serializable() decorator "announces" the custom element to keras.
# This allows keras to find them when we load a model from the disk that uses those custom elements.
# Without this decorator, loading a model that uses one of these custom element would trigger an error.


@tf.keras.utils.register_keras_serializable()
class ClipLayer(keras.layers.Layer):
    """
    Layer that clips values between -200 and 200.

    The output tensor has the same shape as the input tensor.

    If a value is < -200, it is set to -200. If a value is > 200, it is set to 200. Otherwise, it is left unmodified.
    """

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.clip_by_value(inputs, -200, 200)


@tf.keras.utils.register_keras_serializable()
class RoundLayer(keras.layers.Layer):
    """
    Layer that rounds values to the closest integer.

    The output tensor has the same shape as the input tensor.
    """

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.math.round(inputs)


@tf.keras.utils.register_keras_serializable()
class GrayscaleToRGBLayer(keras.layers.Layer):
    """
    Layer that converts an input tensor with shape (<batch size>, <dim1>, <dim2>, 1) to an output tensor with
    shape (<batch size>, <dim1>, <dim2>, 3) by duplicating the values.
    """
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.image.grayscale_to_rgb(inputs)


@tf.keras.utils.register_keras_serializable()
def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor) -> tf.Tensor:
    """
    Dice coefficient metric.

    `true_masks` and `model_outputs` have shape (<batch size>, 512, 512, 1).

    Returns a 1D tf tensor of shape (<batch size>) containing the computed dice coefficients.
    """

    # the import has to be done here to avoid cyclic imports
    import final_model

    predicted_masks = final_model.post_processing_model(model_outputs)
    true_masks_2d = tf.reshape(true_masks, shape=(-1, 512, 512))
    return model_evaluation.dice_coefficients_between_mask_batches(predicted_masks, true_masks_2d)
