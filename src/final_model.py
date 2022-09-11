"""
Code for training the final model and for using it to perform segmentations.

The final model takes an input of shape (<batch size>, 512, 512) where each element represents a pixel value.
The output has shape (<batch size>, 512, 512) where each element is either 0 or 1.
"""
import keras
import keras.layers
import keras_tuner
import numpy as np
import scipy.signal
import skimage.transform
import tensorflow as tf

import config
import custom_keras_objects
import data.preloaded.load
import rootdir
import utils.mask_processing
from model_training import build_model, train_model

MODEL_PATH = rootdir.PROJECT_ROOT_PATH / "model"


def train_final_model() -> None:
    """
    Trains the final model on the whole training dataset. The model is saved on the disk in <project rooot> / model

    The hyperparameter values are determined by the `default` values provided in the `model_training` module.
    """
    hp = keras_tuner.HyperParameters()
    model = build_model(hp)
    train_dataset = data.preloaded.load.train_tf_dataset(shuffle=True)
    train_model(hp, model=model, train_dataset=train_dataset)

    # The trained model has an input shape of (512, 512, 1) and an output shape of (512, 512, 1) that contains
    # probability values between 0 and 1.
    # We add preprocessing and postprocessing layers to make the final model take (512, 512) input shape and to
    # have an output of shape (512, 512) that contains values rounded to 0 or 1.
    final_model = keras.Sequential()
    final_model.add(keras.Input(shape=(512, 512)))
    final_model.add(keras.layers.Reshape(target_shape=(512, 512, 1)))
    final_model.add(model)
    final_model.add(post_processing_model)
    final_model.compile()

    final_model.save(MODEL_PATH)


def _build_post_processing_model() -> keras.Model:
    """
    Returns a model that takes an input with shape (512, 512, 1) containing values between 0 and 1
    and that returns an output with shape (512, 512) and values rounded to either 0 or 1.
    """
    post_processing_model_ = keras.Sequential()
    post_processing_model_.add(keras.Input(shape=(512, 512, 1)))
    post_processing_model_.add(custom_keras_objects.RoundLayer())
    post_processing_model_.add(keras.layers.Reshape(target_shape=(512, 512)))
    return post_processing_model_


post_processing_model = _build_post_processing_model()


def predict(images: tf.data.Dataset) -> np.ndarray:
    """
    Uses the final model stored on the disk to compute segmentation masks for a batch of images.

    Each element of `images` is a 512 x 512 tensor corresponding to an image.

    After the model prediction, some postprocessing is applied on the predicted masks to try to slightly improve it.

    Returns a numpy array of shape (<number of images>, 512, 512) containing the predicted masks.
    """
    batched_images = images.batch(config.PREDICTION_BATCH_SIZE)
    model = keras.models.load_model(MODEL_PATH)
    predicted_masks = model.predict(batched_images)

    predicted_masks_np = np.array(predicted_masks)

    post_processed_predicted_masks = np.array(list(map(final_post_processing, predicted_masks_np)))

    return post_processed_predicted_masks


def predict_from_images_iterable(images) -> list[np.ndarray]:
    """
    Higher level and more convenient function for performing segmentations.

    `images` is an iterable of 2D numpy arrays, where each numpy array corresponds to an image.
    The images can have shapes different from (512, 512).

    If an image has a shape different from (512, 512), it is resized to (512, 512), a mask is predicted using the model
    and this mask is resized back to the original resolution of the image.

    Returns a list of numpy 2D arrays where each array is the mask predicted for the corresponding image, with the same
    resolution as this image.
    """
    def resize_image(image):
        if image.shape == (512, 512):
            return image
        else:
            return skimage.transform.resize(image, output_shape=(512, 512), preserve_range=True)

    images_with_correct_shape = np.array(list(map(resize_image, images)))

    masks = predict(tf.data.Dataset.from_tensor_slices(images_with_correct_shape))

    def resize_mask(mask, image):
        if mask.shape == image.shape:
            return mask
        else:
            resized_mask = skimage.transform.resize(mask, output_shape=image.shape, preserve_range=True).round()
            smoothed_resized_mask = scipy.signal.medfilt(resized_mask, 3)
            return smoothed_resized_mask

    return list(map(resize_mask, masks, images))


# those values have been empirically found to give good results
MAX_DISTANCE_BETWEEN_MASK_AREAS = 10
MIN_RATIO_BETWEEN_MAIN_MASK_COMPONENTS = .1
MAXIMUM_SIZE_REMOVE_SMALL_AREAS = 50


def final_post_processing(mask: np.ndarray) -> np.ndarray:
    """
    Slightly improves a mask outputted by the model by trying to remove some isolated incorrect areas.
    """
    mask = utils.mask_processing.remove_isolated_areas(mask,
                                                       min_ratio=MIN_RATIO_BETWEEN_MAIN_MASK_COMPONENTS,
                                                       max_distance=MAX_DISTANCE_BETWEEN_MASK_AREAS)
    mask = utils.mask_processing.remove_small_areas(mask, MAXIMUM_SIZE_REMOVE_SMALL_AREAS)
    return mask
