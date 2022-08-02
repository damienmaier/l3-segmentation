import functools

import keras
import keras.layers
import keras_tuner
import tensorflow as tf

import architectures
import config
import custom_layers
import model_evaluation
import utils.functional


def build_model(hp: keras_tuner.HyperParameters):
    architecture_name = hp.Choice("architecture", ["deeplabv3"], default="deeplabv3")
    base_model = architectures.architecture_builders[architecture_name]()
    return base_model


def train_model(hp: keras_tuner.HyperParameters, base_model: keras.Model,
                train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
                *args, **kwargs):
    train_dataset = _prepare_dataset_for_training(train_dataset, batch_size=config.TRAINING_BATCH_SIZE,
                                                  is_validation_dataset=False, hp=hp)

    if validation_dataset is not None:
        validation_dataset = _prepare_dataset_for_training(validation_dataset, batch_size=config.TRAINING_BATCH_SIZE,
                                                           is_validation_dataset=True, hp=hp)

    final_model = keras.Sequential()
    final_model.add(keras.Input(shape=(512, 512, 1)))
    if hp.Boolean("clip preprocessing", default=True):
        final_model.add(custom_layers.ClipLayer())
    final_model.add(base_model)

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-3,
        sampling="log",
        default=1e-4
    )

    def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor):
        predicted_masks = post_processing_model(model_outputs)
        true_masks_2d = tf.reshape(true_masks, shape=(-1, 512, 512))
        return model_evaluation.dice_coefficients_between_multiple_pairs_of_masks(predicted_masks, true_masks_2d)

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
        epochs=4,
        *args, **kwargs
    )

    return history


def _prepare_dataset_for_training(dataset: tf.data.Dataset, batch_size: int, is_validation_dataset: bool,
                                  hp: keras_tuner.HyperParameters) -> tf.data.Dataset:
    add_color_axis = functools.partial(tf.reshape, shape=(512, 512, 1))
    dataset = dataset.map(utils.functional.function_on_pair(add_color_axis))

    if not is_validation_dataset:
        dataset = _perform_data_augmentation(dataset, hp)

    if hp.Boolean("weighted loss", default=False):
        dataset = dataset.map(_add_pixel_weights)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def _perform_data_augmentation(dataset: tf.data.Dataset, hp: keras_tuner.HyperParameters):
    if hp.Boolean("horizontal flip", default=True):
        def random_left_right_flip(image, mask):
            seed = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
            transformed_image = tf.image.stateless_random_flip_left_right(image, seed)
            transformed_mask = tf.image.stateless_random_flip_left_right(mask, seed)
            return transformed_image, transformed_mask

        dataset = dataset.map(random_left_right_flip)

    def gaussian_noise(image, mask):
        gaussian_noise_standard_deviation = hp.Float("gaussian noise", min_value=1e-2, max_value=1e2, default=5, sampling="log")
        gaussian_noise_layer = keras.layers.GaussianNoise(gaussian_noise_standard_deviation)
        return gaussian_noise_layer(image, training=True), mask

    dataset = dataset.map(gaussian_noise)
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


post_processing_model = keras.Sequential()
post_processing_model.add(keras.Input(shape=(512, 512, 1)))
post_processing_model.add(custom_layers.RoundLayer())
post_processing_model.add(keras.layers.Reshape(target_shape=(512, 512)))
