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


def train_final_model():
    hp = keras_tuner.HyperParameters()
    model = build_model(hp)
    train_dataset = data.preloaded.load.train_tf_dataset(shuffle=True)
    train_model(hp, model=model, train_dataset=train_dataset)

    final_model = keras.Sequential()
    final_model.add(keras.Input(shape=(512, 512)))
    final_model.add(keras.layers.Reshape(target_shape=(512, 512, 1)))
    final_model.add(model)
    final_model.add(post_processing_model)
    final_model.compile()

    final_model.save(MODEL_PATH)


def _build_post_processing_model():
    post_processing_model_ = keras.Sequential()
    post_processing_model_.add(keras.Input(shape=(512, 512, 1)))
    post_processing_model_.add(custom_keras_objects.RoundLayer())
    post_processing_model_.add(keras.layers.Reshape(target_shape=(512, 512)))
    return post_processing_model_


post_processing_model = _build_post_processing_model()


def predict(images: tf.data.Dataset) -> np.ndarray:
    batched_images = images.batch(config.PREDICTION_BATCH_SIZE)
    model = keras.models.load_model(MODEL_PATH)
    predicted_masks = model.predict(batched_images)

    predicted_masks_np = np.array(predicted_masks)

    post_processed_predicted_masks = np.array(list(map(final_post_processing, predicted_masks_np)))

    return post_processed_predicted_masks


def predict_from_images_iterable(images) -> list[np.ndarray]:
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
    mask = utils.mask_processing.remove_isolated_areas(mask,
                                                       min_ratio=MIN_RATIO_BETWEEN_MAIN_MASK_COMPONENTS,
                                                       max_distance=MAX_DISTANCE_BETWEEN_MASK_AREAS)
    mask = utils.mask_processing.remove_small_areas(mask, MAXIMUM_SIZE_REMOVE_SMALL_AREAS)
    return mask
