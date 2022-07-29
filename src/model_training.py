import pathlib

import keras
import keras_tuner
import numpy as np
import sklearn.model_selection
import tensorflow as tf

import architectures._unet
import config
import custom_layers
import dataset.data_loading
import model_evaluation


def build_model(hp: keras_tuner.HyperParameters):

    architecture_name = hp.Choice("architecture", ["unet", "deeplabv3"], default="unet")
    base_model = architectures.architecture_builders[architecture_name]()

    if hp.Boolean("clip preprocessing", default=False):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(512, 512)))
        model.add(custom_layers.ClipLayer())
        model.add(base_model)
    else:
        model = base_model

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-2,
        sampling="log",
        default=1e-4
    )

    def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor):
        predicted_masks = custom_layers.RoundLayer()(model_outputs)
        return model_evaluation.dice_coefficients_between_multiple_pairs_of_masks(true_masks, predicted_masks)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=dice,
        # avoid tf complaining from the fact that we give pixel weights without having a weighted metric
        weighted_metrics=[]
    )

    return model


def train_model(hp: keras_tuner.HyperParameters, model: keras.Model,
                images_paths: list[pathlib.Path], masks_paths: list[pathlib.Path], use_validation_set: bool,
                *args, **kwargs):
    if use_validation_set:
        images_paths_train, images_paths_validation, masks_paths_train, masks_paths_validation = \
            sklearn.model_selection.train_test_split(images_paths, masks_paths)
    else:
        images_paths_train = images_paths
        masks_paths_train = masks_paths

    use_weighted_loss = hp.Boolean("weighted loss", default=False)

    train_dataset = _create_tf_dataset_for_training(images_paths_train, masks_paths_train,
                                                    batch_size=config.TRAINING_BATCH_SIZE,
                                                    add_pixel_weights=use_weighted_loss)

    if use_validation_set:
        validation_dataset = _create_tf_dataset_for_training(images_paths_validation, masks_paths_validation,
                                                             batch_size=config.PREDICTION_BATCH_SIZE,
                                                             add_pixel_weights=use_weighted_loss)
    else:
        validation_dataset = None

    return model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=5,
        *args, **kwargs
    )


def _create_tf_dataset_for_training(images_paths: list[pathlib.Path], masks_paths: list[pathlib.Path],
                                    batch_size: int, add_pixel_weights: bool) -> tf.data.Dataset:
    images_paths_str_array = np.array(list(map(str, images_paths)))
    masks_paths_str_array = np.array(list(map(str, masks_paths)))
    paths_dataset = tf.data.Dataset.from_tensor_slices((images_paths_str_array, masks_paths_str_array,))

    shuffled_paths_dataset = paths_dataset.shuffle(buffer_size=paths_dataset.cardinality())

    def load_tf_tensors_from_files(image_path: str, mask_path: str):
        return dataset.data_loading.load_tf_tensor_from_file(image_path), dataset.data_loading.load_tf_tensor_from_file(
            mask_path)

    base_dataset = shuffled_paths_dataset.map(load_tf_tensors_from_files)

    if add_pixel_weights:
        base_dataset = base_dataset.map(_add_pixel_weights)

    batched_dataset = base_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return batched_dataset


def _add_pixel_weights(image: tf.Tensor, mask: tf.Tensor):
    mask_0_count = tf.cast(tf.math.count_nonzero(mask == 0), tf.float64)
    mask_1_count = tf.cast(tf.math.count_nonzero(mask), tf.float64)
    mask_size = tf.cast(tf.size(mask), tf.float64)

    weight_for_class_0 = mask_size / 2. / mask_0_count
    weight_for_class_1 = mask_size / 2. / mask_1_count
    class_weights = tf.stack([weight_for_class_0, weight_for_class_1])

    pixel_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

    # for an unknown reason the weights must have this shape,
    # even if the images and masks have shape (512, 512)
    pixel_weights = tf.reshape(pixel_weights, shape=(512, 512, 1))

    return image, mask, pixel_weights
