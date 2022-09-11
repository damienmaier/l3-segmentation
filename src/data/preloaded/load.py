"""
Module for reading data from the preloaded dataset

The functions for accessing the dataset return tf dataset objects.
The data is not loaded in memory when these functions are executed. When an element of a tf dataset
is accessed, the corresponding data is red from the disk.

This allows to train the model on large datasets that do not fit in memory.
"""

import pathlib

import numpy as np
import sklearn.model_selection
import tensorflow as tf

import utils.functional
from data.preloaded import TRAIN_SET_PATH, TEST_SET_PATH, IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, PREDICTIONS_FOLDER_NAME


def train_validation_tf_datasets(random_state=None) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Provides a random split of the train set into a train and a validation set.

    The preloaded dataset is made of a training set and a test set. This function provides a "training" and a validation
    set that only contains data from the training set of the preloaded dataset. This function does not give access
    to data from the test set of the preloaded dataset.

    `random_state` is the seed used for the random split. If this function is called several times with the same value
    for `random state`, il will always return the same split.`

    Returns
        - A tf dataset for the training data. Each element of the tf dataset is an (image, mask) pair.
        - A tf dataset for the validation data. Each element of the tf dataset is an (image, mask) pair.
    """
    images_paths, masks_paths = _train_set_paths()
    images_paths_train, images_paths_validation, masks_paths_train, masks_paths_validation = \
        sklearn.model_selection.train_test_split(images_paths, masks_paths, random_state=random_state)

    train_dataset = _tf_dataset_from_images_masks_paths(images_paths_train, masks_paths_train, shuffle=True)
    validation_dataset = _tf_dataset_from_images_masks_paths(images_paths_validation, masks_paths_validation,
                                                             shuffle=True)
    return train_dataset, validation_dataset


def train_tf_dataset(shuffle: bool) -> tf.data.Dataset:
    """
    Returns a tf dataset for the training data. Each element of the tf dataset is an (image, mask) pair.

    If `shuffle` is set to true, the returned tf dataset will provide the data in a random order.
    """
    return _tf_dataset_from_images_masks_paths(*_train_set_paths(), shuffle=shuffle)


def test_tf_dataset(shuffle: bool) -> tf.data.Dataset:
    """
    Returns a tf dataset for the test data. Each element of the tf dataset is an (image, mask) pair.

    If `shuffle` is set to true, the returned tf dataset will provide the data in a random order.
    """
    return _tf_dataset_from_images_masks_paths(*_test_set_paths(), shuffle=shuffle)


def test_images_masks_predictions_tf_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Returns
        - A tf dataset for the images of the test set
        - A tf dataset for the true masks of the test set
        - A tf dataset for the predicted masks of the test set
    """
    images_paths, masks_paths, predictions_paths = \
        _get_dataset_files_paths(TEST_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, PREDICTIONS_FOLDER_NAME])
    images_dataset = _tf_dataset_from_paths(images_paths)
    masks_dataset = _tf_dataset_from_paths(masks_paths)
    predictions_dataset = _tf_dataset_from_paths(predictions_paths)
    return images_dataset, masks_dataset, predictions_dataset


def _tf_dataset_from_paths(paths: list[str]) -> tf.data.Dataset:
    """
    `paths` is a list of paths to files on the disk that each contain a tf tensor.

    Returns a tf dataset where each element is one of those tf tensors.
    """
    return tf.data.Dataset.from_tensor_slices(paths).map(_load_tf_tensor_from_file)


def _tf_dataset_from_images_masks_paths(images_paths: np.ndarray, masks_paths: np.ndarray,
                                        shuffle: bool) -> tf.data.Dataset:
    """
    `images_paths` and `masks_paths` are 1D str numpy arrays that contain paths of tf tensor stored on the disk.

    Returns a tf dataset where each element is an (image, mask) pair.
    """
    paths_dataset = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))
    if shuffle:
        paths_dataset = paths_dataset.shuffle(buffer_size=paths_dataset.cardinality())
    dataset = paths_dataset.map(utils.functional.function_on_pair(_load_tf_tensor_from_file))
    return dataset


def _train_set_paths() -> list[np.ndarray, np.ndarray]:
    return _get_dataset_files_paths(TRAIN_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME])


def _test_set_paths() -> list[np.ndarray, np.ndarray]:
    return _get_dataset_files_paths(TEST_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME])


def _load_tf_tensor_from_file(file_path: str) -> tf.Tensor:
    serialized_tensor = tf.io.read_file(file_path)
    tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.dtypes.float64)
    return tensor


def _get_dataset_files_paths(dataset_path: pathlib.Path, sub_folder_names) -> list[np.ndarray]:
    """
    Computes the paths of dataset files. If `sub_folder_names` = ["a", "b", "c"], returns the paths of the files
    in `dataset_path` / "a", `dataset_path` / "b" and `dataset_path` / "c"

    Checks that the files in each sub-folders have the same names.

    For each element of `sub_folder_names`, returns the list of file paths as a 1D str numpy array.
    """
    dataset_sub_folders_paths = [dataset_path / sub_folder_name for sub_folder_name in sub_folder_names]
    file_paths_for_each_folder = list(map(_sorted_file_paths_of_directory, dataset_sub_folders_paths))

    def two_files_have_same_name(file1_path: pathlib.Path, file2_path: pathlib.Path):
        return file1_path.name == file2_path.name

    def files_have_same_name(files_paths_list):
        first_file_path = files_paths_list[0]
        return all(two_files_have_same_name(first_file_path, other_file_path) for other_file_path in files_paths_list)

    files_in_each_folder_have_the_same_names = all(map(files_have_same_name, zip(*file_paths_for_each_folder)))
    assert files_in_each_folder_have_the_same_names

    def convert_paths_to_str_array(paths: list[pathlib.Path]) -> np.ndarray:
        paths_str = list(map(str, paths))
        return np.array(paths_str)

    return list(map(convert_paths_to_str_array, file_paths_for_each_folder))


def _sorted_file_paths_of_directory(directory_path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory_path.iterdir())
