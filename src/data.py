import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetElement:
    dataset_subdirectory_path: Path
    file_name: str

    def __repr__(self) -> str:
        return f"path {self.dataset_subdirectory_path}, file name {self.file_name}"


def read_dataset(dataset_path: Path):
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
    The files must be CSV files containing an array of values separated by commas.

    Returns a list of images and a list of corresponding masks. The elements of the lists are functions that,
    when called, will read the corresponding file on the disk and return the image or mask matrix as a numpy array.
    """

    dataset_subdirectory_paths = dataset_path.iterdir()
    dataset_elements = itertools.chain(*map(dataset_elements_from_subdirectory, dataset_subdirectory_paths))
    element_file_paths = [(dataset_element.dataset_subdirectory_path / "images" / dataset_element.file_name,
                           dataset_element.dataset_subdirectory_path / "masks" / dataset_element.file_name)
                          for dataset_element in dataset_elements]
    images_paths, masks_paths = zip(*element_file_paths)
    images_functions = map(get_function_that_returns_csv_file_content, images_paths)
    masks_functions = map(get_function_that_returns_csv_file_content, masks_paths)

    return list(images_functions), list(masks_functions)


def dataset_elements_from_subdirectory(subdirectory_path: Path):
    for file_path in (subdirectory_path / "images").iterdir():
        yield DatasetElement(subdirectory_path, file_path.name)


def get_function_that_returns_csv_file_content(csv_file_path: Path):
    return lambda: np.loadtxt(csv_file_path, delimiter=",")
