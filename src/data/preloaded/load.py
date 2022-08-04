import pathlib

import numpy as np
import sklearn.model_selection
import tensorflow as tf

import utils.functional
from data.preloaded import TRAIN_SET_PATH, TEST_SET_PATH, IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, PREDICTIONS_FOLDER_NAME


def train_validation_tf_datasets(random_state=None):
    images_paths, masks_paths = _train_set_paths()
    images_paths_train, images_paths_validation, masks_paths_train, masks_paths_validation = \
        sklearn.model_selection.train_test_split(images_paths, masks_paths, random_state=random_state)

    train_dataset = _tf_dataset_from_images_masks_paths(images_paths_train, masks_paths_train, shuffle=True)
    validation_dataset = _tf_dataset_from_images_masks_paths(images_paths_validation, masks_paths_validation,
                                                             shuffle=True)
    return train_dataset, validation_dataset


def train_tf_dataset(shuffle: bool):
    return _tf_dataset_from_images_masks_paths(*_train_set_paths(), shuffle=shuffle)


def test_tf_dataset(shuffle: bool):
    return _tf_dataset_from_images_masks_paths(*_test_set_paths(), shuffle=shuffle)


def test_images_masks_predictions_tf_datasets():
    images_paths, masks_paths, predictions_paths = \
        _get_dataset_files_paths(TEST_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME, PREDICTIONS_FOLDER_NAME])
    images_dataset = _tf_dataset_from_paths(images_paths)
    masks_dataset = _tf_dataset_from_paths(masks_paths)
    predictions_dataset = _tf_dataset_from_paths(predictions_paths)
    return images_dataset, masks_dataset, predictions_dataset


def _tf_dataset_from_paths(paths: list[str]):
    return tf.data.Dataset.from_tensor_slices(paths).map(_load_tf_tensor_from_file)


def _tf_dataset_from_images_masks_paths(images_paths: np.ndarray, masks_paths: np.ndarray,
                                        shuffle: bool) -> tf.data.Dataset:
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
