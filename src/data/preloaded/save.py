import pathlib
import shutil

import numpy as np
import sklearn.model_selection
import tensorflow as tf

from data.preloaded import PRELOADED_DATASET_PATH, TRAIN_SET_PATH, TEST_SET_PATH, IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, \
    PREDICTIONS_FOLDER_NAME


def preload_original_dataset(images, masks):
    if PRELOADED_DATASET_PATH.exists():
        print("Error : a preloaded dataset already exists")
    else:
        PRELOADED_DATASET_PATH.mkdir()
        images_train, images_test, masks_train, masks_test = \
            sklearn.model_selection.train_test_split(images, masks, random_state=42)

        _save_images_and_masks_on_disk(images_train, masks_train, TRAIN_SET_PATH)
        _save_images_and_masks_on_disk(images_test, masks_test, TEST_SET_PATH)


def save_test_predictions(predictions):
    predictions_directory_path = TEST_SET_PATH / PREDICTIONS_FOLDER_NAME
    if predictions_directory_path.exists():
        shutil.rmtree(predictions_directory_path)
    _save_np_array_as_tf_tensors_on_disk(predictions, predictions_directory_path)


def _save_images_and_masks_on_disk(images, masks, directory_path):
    _save_np_array_as_tf_tensors_on_disk(images, directory_path / IMAGES_FOLDER_NAME)
    _save_np_array_as_tf_tensors_on_disk(masks, directory_path / MASKS_FOLDER_NAME)


def _save_np_array_as_tf_tensors_on_disk(array: np.ndarray, directory_path: pathlib.Path):
    for file_index, array_2d in enumerate(array):
        tensor = tf.convert_to_tensor(array_2d, dtype=tf.dtypes.float64)
        tf.io.write_file(
            filename=str(directory_path / str(file_index)),
            contents=tf.io.serialize_tensor(tensor)
        )
