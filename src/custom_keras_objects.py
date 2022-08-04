import keras.layers
import keras.utils
import tensorflow as tf

import model_evaluation


@tf.keras.utils.register_keras_serializable()
class ClipLayer(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.clip_by_value(inputs, -200, 200)


@tf.keras.utils.register_keras_serializable()
class RoundLayer(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.math.round(inputs)


@tf.keras.utils.register_keras_serializable()
class GrayscaleToRGBLayer(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.image.grayscale_to_rgb(inputs)


@tf.keras.utils.register_keras_serializable()
def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor):
    import final_model
    predicted_masks = final_model.post_processing_model(model_outputs)
    true_masks_2d = tf.reshape(true_masks, shape=(-1, 512, 512))
    return model_evaluation.dice_coefficients_between_mask_batches(predicted_masks, true_masks_2d)
