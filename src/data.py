import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from rootdir import PROJECT_ROOT_PATH

IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"

DATASET_PATH = PROJECT_ROOT_PATH / "dataset"
CACHE_FILE_PATH = PROJECT_ROOT_PATH / "cache" / "dataset.npy"


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

        Loaded data is saved in a cache directory. Subsequent calls to this function will ignore the dataset and load
        from the cache directory, to improve speed.

        Returns:
            The images arrays
            The corresponding masks arrays
    """

    if CACHE_FILE_PATH.exists():
        images, masks = np.load(CACHE_FILE_PATH, allow_pickle=True)
    else:
        images, masks = _read_dataset_from_disk()
        CACHE_FILE_PATH.parent.mkdir(exist_ok=True)
        data_to_save = np.array([images, masks], dtype=object)
        np.save(CACHE_FILE_PATH, data_to_save)

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

    images = map(file_reader, images_paths)
    masks = map(file_reader, masks_paths)

    return list(images), list(masks)
