import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import sklearn.model_selection

from rootdir import PROJECT_ROOT_PATH

IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"

DATASET_PATH = PROJECT_ROOT_PATH / "dataset"
CACHE_FILE_PATH = PROJECT_ROOT_PATH / "cache" / "dataset.npy"


def get_test_set():
    _, X_test, _, y_test = _get_splitted_dataset()
    return X_test, y_test


def get_train_set():
    X_train, _, y_train, _ = _get_splitted_dataset()
    return X_train, y_train


def _get_splitted_dataset():
    return sklearn.model_selection.train_test_split(*load_dataset(), random_state=42)


def load_dataset():
    """
        The dataset directory is expected to have the following structure:
            dataset directory
                subdir1
                    images
                        file1

                        file2
                    masks
                        file1

                        file2
                subdir2
                    images
                        file3

                        file4
                    masks
                        file3

                        file4
                ...
        The files must be CSV files containing a matrix of values separated by commas.
        Matrices must have a 512 x 512 shape. Files containing a matrix with a different shape are ignored.

        Loaded data is saved in a cache directory. Subsequent calls to this function will ignore the dataset and load
        from the cache directory, to improve speed.

        Returns:
            The images arrays
            The corresponding masks arrays
    """
    print("begin loading data")
    if CACHE_FILE_PATH.exists():
        images, masks = np.load(CACHE_FILE_PATH, allow_pickle=True)
    else:
        images, masks = _read_dataset_from_disk()
        CACHE_FILE_PATH.parent.mkdir(exist_ok=True)
        data_to_save = np.stack([images, masks])
        np.save(CACHE_FILE_PATH, data_to_save)

    print("end loading data")
    return images, masks


@dataclass
class DatasetElement:
    """
    Represents an element of a dataset.
    The element has an image file in `dataset_subdirectory_path` / `IMAGES_FOLDER_NAME`
    and a mask file in `dataset_subdirectory_path` / `MASKS_FOLDER_NAME`
    """
    dataset_subdirectory_path: Path
    file_name: str

    def __repr__(self) -> str:
        return f"path {self.dataset_subdirectory_path}, file name {self.file_name}"

    def file_path_string(self, folder_name: str) -> str:
        file_path = self.dataset_subdirectory_path / folder_name / self.file_name
        return str(file_path)

    def image_path_string(self):
        """
        :return: The path string of the image file
        """
        return self.file_path_string(IMAGES_FOLDER_NAME)

    def mask_path_string(self):
        """
        :return: The path string of the mask file
        """
        return self.file_path_string(MASKS_FOLDER_NAME)


def _read_dataset_from_disk():
    dataset_subdirectory_paths = DATASET_PATH.iterdir()

    def dataset_elements_from_subdirectory(subdirectory_path: Path) -> Generator[DatasetElement, None, None]:
        """
        generator that yields all dataset elements of a dataset subdirectory
        """
        for file_path in (subdirectory_path / IMAGES_FOLDER_NAME).iterdir():
            yield DatasetElement(subdirectory_path, file_path.name)

    dataset_elements = list(itertools.chain(*map(dataset_elements_from_subdirectory, dataset_subdirectory_paths)))

    images_paths = map(DatasetElement.image_path_string, dataset_elements)
    masks_paths = map(DatasetElement.mask_path_string, dataset_elements)

    def file_reader(csv_file_path: str):
        return np.loadtxt(csv_file_path, delimiter=",")

    def dataset_element_files_reader(image_path: str, mask_path: str):
        return file_reader(image_path), file_reader(mask_path)

    dataset_data = map(dataset_element_files_reader, images_paths, masks_paths)

    def dataset_element_has_correct_shape(image_mask_tuple):
        image, mask = image_mask_tuple
        shape_is_correct = image.shape == (512, 512) and mask.shape == (512, 512)
        if not shape_is_correct:
            print("A dataset element was ignored because its shape is not 512 x 512")
        return shape_is_correct

    filtered_dataset_data = filter(dataset_element_has_correct_shape, dataset_data)

    images, masks = zip(*filtered_dataset_data)

    return np.array(images), np.array(masks)
