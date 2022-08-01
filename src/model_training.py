import keras
import keras.layers
import keras_tuner
import sklearn.model_selection
import tensorflow as tf

import architectures
import config
import custom_layers
import data.preloaded.load
import model_evaluation
import utils.functional


def build_model(hp: keras_tuner.HyperParameters):
    architecture_name = hp.Choice("architecture", ["unet", "deeplabv3"], default="deeplabv3")
    base_model = architectures.architecture_builders[architecture_name]()
    return base_model


def train_model(hp: keras_tuner.HyperParameters, base_model: keras.Model,
                train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                *args, **kwargs):
    use_weighted_loss = hp.Boolean("weighted loss", default=False)
    train_dataset = _prepare_dataset_for_training(train_dataset,
                                                  batch_size=config.TRAINING_BATCH_SIZE,
                                                  add_pixel_weights=use_weighted_loss)

    if validation_dataset is not None:
        validation_dataset = _prepare_dataset_for_training(validation_dataset,
                                                           batch_size=config.PREDICTION_BATCH_SIZE,
                                                           add_pixel_weights=use_weighted_loss)

    final_model = keras.Sequential()
    final_model.add(base_model.input)

    if hp.Boolean("clip preprocessing", default=True):
        final_model.add = custom_layers.ClipLayer()

    if hp.Boolean("data normalization", default=True):
        normalization_layer = keras.layers.Normalization(axis=None)
        normalization_layer.adapt(train_dataset)
        final_model.add(normalization_layer)

    final_model.add(base_model)

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-2,
        sampling="log",
        default=1e-4
    )

    def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor):

        predicted_masks = custom_layers.RoundLayer()(model_outputs)
        predicted_masks_2d = tf.reshape(predicted_masks, shape=(-1, 512, 512))
        true_masks_2d = tf.reshape(true_masks, shape=(-1, 512, 512))
        return model_evaluation.dice_coefficients_between_multiple_pairs_of_masks(predicted_masks_2d, true_masks_2d)

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=dice,
        # avoid tf complaining from the fact that we give pixel weights without having a weighted metric
        weighted_metrics=[]
    )

    history = final_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=5,
        *args, **kwargs
    )

    return history


def _prepare_dataset_for_training(dataset: tf.data.Dataset, batch_size: int,
                                  add_pixel_weights: bool) -> tf.data.Dataset:
    dataset = dataset.map(utils.functional.function_on_pair(keras.layers.Reshape(target_shape=(512, 512, 1))))
    if add_pixel_weights:
        dataset = dataset.map(_add_pixel_weights)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def _add_pixel_weights(image: tf.Tensor, mask: tf.Tensor):
    mask_0_count = tf.cast(tf.math.count_nonzero(mask == 0), tf.float64)
    mask_1_count = tf.cast(tf.math.count_nonzero(mask), tf.float64)
    mask_size = tf.cast(tf.size(mask), tf.float64)

    weight_for_class_0 = mask_size / 2. / mask_0_count
    weight_for_class_1 = mask_size / 2. / mask_1_count
    class_weights = tf.stack([weight_for_class_0, weight_for_class_1])

    pixel_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

    return image, mask, pixel_weights
