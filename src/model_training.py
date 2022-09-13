"""
Provides functions to train the model.

The `hp` argument of the functions allows to declare and to configure hyperparameters.
When the model hyperparameters exploration is done by the `model_exploration` module, hyperparameter values
are chosen according to the `min_value` and `max_value` of each hyperparameter. When the final model is trained,
the values of the hyperparameters are determined by the `default` value of the hyperparameters.

You can therefore control the hyperparameter ranges for the model exploration and the hyperparameter final values
for the final model training by modifying `min_value`, `max_value` and `default`.
"""

import functools

import keras
import keras.callbacks
import keras.layers
import keras_tuner
import tensorflow as tf

import architectures
import config
import custom_keras_objects
import utils.data_augmentation
import utils.display_image
import utils.functional


def build_model(hp: keras_tuner.HyperParameters) -> keras.Model:
    """
    Builds a tf model.

    Hyperparameters:
        - Architecture : controls which model architecture is used. The architectures are defined in the `architectures` module.
        - Clip preprocessing : if true, a preprocessing layer is added that clips the input pixel values in the
        [-200, 200] range.
    """
    architecture_name = hp.Choice("architecture", ["deeplabv3", "unet"], default="deeplabv3")
    base_model = architectures.architecture_builders[architecture_name]()

    final_model = keras.Sequential()
    final_model.add(keras.Input(shape=(512, 512, 1)))
    if hp.Boolean("clip preprocessing", default=True):
        final_model.add(custom_keras_objects.ClipLayer())
    final_model.add(base_model)

    return final_model


def train_model(hp: keras_tuner.HyperParameters, model: keras.Model,
                train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset = None,
                *args, **kwargs) -> keras.callbacks.History:
    """
    Trains `model` using `train_dataset`. At each epoch, the performance is evaluated on `validation_dataset`.

    Hyperparameters:
        - weighted loss : if True, uses a weighted loss function
        - horizontal flip : if True, a horizontal flip with 1/2 probability is applied on the training data each time it is shown to the model
        - rotation : if > 0, a random rotation is applied on the training data each time it is shown to the model. This value controls the max angle of the random rotation.
        - gaussian noise : if > 0, a random gaussian noise is applied on the training data each time it is shown to the model. This value controls the standard deviation for the gaussian noise.

    Extra arguments are passed to the `fit` method of the model.

    Returns the history object returned by the `fit` method of the model.
    """
    train_dataset = _prepare_dataset_for_training(train_dataset, batch_size=config.TRAINING_BATCH_SIZE,
                                                  is_validation_dataset=False, hp=hp)

    if validation_dataset is not None:
        validation_dataset = _prepare_dataset_for_training(validation_dataset, batch_size=config.TRAINING_BATCH_SIZE,
                                                           is_validation_dataset=True, hp=hp)

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-3,
        default=2e-4
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=custom_keras_objects.dice,
        # avoid tf complaining from the fact that we give pixel weights without having a weighted metric
        weighted_metrics=[]
    )

    history = model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        *args, **kwargs
    )

    return history


def _prepare_dataset_for_training(dataset: tf.data.Dataset, batch_size: int, is_validation_dataset: bool,
                                  hp: keras_tuner.HyperParameters) -> tf.data.Dataset:
    """
    Adds several transformations to the pipeline of the dataset

    - Each image and mask is reshaped from (512, 512) to (512, 512, 1). This is necessary because this is the shape expected by the model architectures.
    - Data augmentation is performed if `is_validation_dataset` is True
    - If the `weighted loss´ hyperparameter is True, an array of pixel weights is added to each dataset element
    - The dataset is batched with respect to `batch_size`
    """
    add_color_axis = functools.partial(tf.reshape, shape=(512, 512, 1))
    dataset = dataset.map(utils.functional.function_on_pair(add_color_axis))

    if not is_validation_dataset:
        dataset = _perform_data_augmentation(dataset, hp)

    if hp.Boolean("weighted loss", default=False):
        dataset = dataset.map(_add_pixel_weights)

    # According to https://www.tensorflow.org/guide/data_performance
    # prefetch increases the performance
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def _perform_data_augmentation(dataset: tf.data.Dataset, hp: keras_tuner.HyperParameters) -> tf.data.Dataset:
    if hp.Boolean("horizontal flip", default=False):
        def random_left_right_flip(image, mask):
            seed = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
            transformed_image = tf.image.stateless_random_flip_left_right(image, seed)
            transformed_mask = tf.image.stateless_random_flip_left_right(mask, seed)
            return transformed_image, transformed_mask

        dataset = dataset.map(random_left_right_flip)

    rotation_angle = hp.Float("rotation", min_value=0, max_value=2, default=0)
    if rotation_angle != 0:
        dataset = dataset.map(utils.data_augmentation.generate_random_rotation_function(max_angle=rotation_angle))

    gaussian_noise_standard_deviation = hp.Float("gaussian noise", min_value=.1, max_value=30, default=0,
                                                 sampling="log")
    if gaussian_noise_standard_deviation != 0:
        def gaussian_noise(image, mask):
            gaussian_noise_layer = keras.layers.GaussianNoise(gaussian_noise_standard_deviation)
            return gaussian_noise_layer(image, training=True), mask

        dataset = dataset.map(gaussian_noise)

    return dataset


def _add_pixel_weights(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Takes as input the tensors for an image and a mask, and computes a weights tensor to be used by a weighted loss
    function.

    Returns
        - The same image tensor
        - The same mask tensor
        - The weights tensor
    """
    # This code was inspired by https://www.tensorflow.org/tutorials/images/segmentation#optional_imbalanced_classes_and_class_weights
    # Quite surprisingly, it makes the model perform worse. There may be an error in this implementation.
    mask_0_count = tf.cast(tf.math.count_nonzero(mask == 0), tf.float64)
    mask_1_count = tf.cast(tf.math.count_nonzero(mask), tf.float64)
    mask_size = tf.cast(tf.size(mask), tf.float64)

    weight_for_class_0 = mask_size / 2. / mask_0_count
    weight_for_class_1 = mask_size / 2. / mask_1_count
    class_weights = tf.stack([weight_for_class_0, weight_for_class_1])

    pixel_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

    return image, mask, pixel_weights


def visualize_prepared_dataset(*args, **kwargs):
    """
    Displays the images and masks from the first batch of the dataset returned by `_prepare_dataset_for_training`.
    This function is used for debugging. It can be useful for visualizing the data augmentation results.
    """
    dataset = _prepare_dataset_for_training(*args, **kwargs)
    images_batch, masks_batch = next(iter(dataset))
    for image, mask in zip(images_batch, masks_batch):
        utils.display_image.display_ct_scan_image_and_mask(tf.squeeze(image), tf.squeeze(mask))
