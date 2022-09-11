"""
Module for writing data to the preloaded dataset
"""

import pathlib
import shutil

import sklearn.model_selection
import tensorflow as tf

from data.preloaded import PRELOADED_DATASET_PATH, TRAIN_SET_PATH, TEST_SET_PATH, IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, \
    PREDICTIONS_FOLDER_NAME


def create_preloaded_dataset(images, masks) -> None:
    """
    Creates the preloaded dataset by writing the images and masks in the appropriate format on the disk.
    The data is split into a train test and a test set.

    :param images: an iterable that contains a 2D numpy array for each image
    :param masks: an iterable that contains a 2D numpy array for each corresponding mask
    """
    if PRELOADED_DATASET_PATH.exists():
        shutil.rmtree(PRELOADED_DATASET_PATH)

    PRELOADED_DATASET_PATH.mkdir()
    images_train, images_test, masks_train, masks_test = \
        sklearn.model_selection.train_test_split(images, masks, random_state=42)

    _save_images_and_masks_on_disk(images_train, masks_train, TRAIN_SET_PATH)
    _save_images_and_masks_on_disk(images_test, masks_test, TEST_SET_PATH)


def save_test_predictions(predictions) -> None:
    """
    Saves the masks predicted by the model for the test images on the preloaded dataset.

    `predictions` is an iterable that contains a 2D numpy array for each predicted mask.
    """
    predictions_directory_path = TEST_SET_PATH / PREDICTIONS_FOLDER_NAME
    if predictions_directory_path.exists():
        shutil.rmtree(predictions_directory_path)
    _save_arrays_as_tf_tensors_on_disk(predictions, predictions_directory_path)


def _save_images_and_masks_on_disk(images, masks, directory_path):
    _save_arrays_as_tf_tensors_on_disk(images, directory_path / IMAGES_FOLDER_NAME)
    _save_arrays_as_tf_tensors_on_disk(masks, directory_path / MASKS_FOLDER_NAME)


def _save_arrays_as_tf_tensors_on_disk(arrays, directory_path: pathlib.Path) -> None:
    """
    `arrays` is an iterable that contains numpy arrays.
    Each array is converted to a tf tensor and saved in an individual file in `directory_path`
    """
    for file_index, array_2d in enumerate(arrays):
        tensor = tf.convert_to_tensor(array_2d, dtype=tf.dtypes.float64)
        tf.io.write_file(
            # Here it is important to left fill the names with 0s
            # Otherwise the lexicographic order of the file names would be different from the index order
            # and the files would be red in a different order than when written
            filename=str(directory_path / str(file_index).zfill(5)),
            contents=tf.io.serialize_tensor(tensor)
        )
