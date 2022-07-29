import pathlib

import numpy as np
import sklearn.model_selection
import tensorflow as tf

from rootdir import PROJECT_ROOT_PATH

PRELOADED_DATASET_PATH = PROJECT_ROOT_PATH / "preloaded dataset"
TRAIN_SET_PATH = PRELOADED_DATASET_PATH / "train set"
TEST_SET_PATH = PRELOADED_DATASET_PATH / "test set"

IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"


def get_train_set():
    return _get_dataset_files_paths(TRAIN_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME])


def get_test_set():
    return _get_dataset_files_paths(TEST_SET_PATH, [IMAGES_FOLDER_NAME, MASKS_FOLDER_NAME])


def load_tf_tensor_from_file(file_path: str) -> tf.Tensor:
    serialized_tensor = tf.io.read_file(file_path)
    tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.dtypes.float64)
    return tensor


def get_tf_tensor_from_tensor_file_paths(tensor_file_paths) -> tf.data.Dataset:
    tensor_file_paths_str = map(str, tensor_file_paths)
    tensors = list(map(load_tf_tensor_from_file, tensor_file_paths_str))
    return tf.stack(tensors)


def preload_original_dataset(images, masks):
    if PRELOADED_DATASET_PATH.exists():
        print("Error : a preloaded dataset already exists")
    else:
        PRELOADED_DATASET_PATH.mkdir()
        images_train, images_test, masks_train, masks_test = \
            sklearn.model_selection.train_test_split(images, masks, random_state=42)

        _save_images_and_masks_on_disk(images_train, masks_train, TRAIN_SET_PATH)
        _save_images_and_masks_on_disk(images_test, masks_test, TEST_SET_PATH)


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


def _sorted_file_paths_of_directory(directory_path: pathlib.Path):
    return sorted(directory_path.iterdir())


def _get_dataset_files_paths(dataset_path: pathlib.Path, sub_folder_names):
    dataset_sub_folders_paths = [dataset_path / sub_folder_name for sub_folder_name in sub_folder_names]
    file_paths_for_each_folder = list(map(_sorted_file_paths_of_directory, dataset_sub_folders_paths))

    def two_files_have_same_name(file1_path: pathlib.Path, file2_path: pathlib.Path):
        return file1_path.name == file2_path.name

    def files_have_same_name(files_paths_list):
        first_file_path = files_paths_list[0]
        return all(two_files_have_same_name(first_file_path, other_file_path) for other_file_path in files_paths_list)

    files_in_each_folder_have_the_same_names = all(map(files_have_same_name, zip(*file_paths_for_each_folder)))
    assert files_in_each_folder_have_the_same_names

    return file_paths_for_each_folder
