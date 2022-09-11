"""
Code for reading the original dataset.

The original dataset is expected to have the following location and structure :
<root directory of the project>
    original dataset
        <subdir1>
            images
                <file1>
                <file2>
                ...
            masks
                <file1>
                <file2>
                ...
        <subdir2>
            images
                <file3>
                <file4>
                ...
            masks
                <file3>
                <file4>
                ...
        ...

Where the directories `original dataset` `images` and `masks` have these exact names.

The names of <subdir1>, <subdir2>, etc. is not significant, and it is not an information used by the code.
It does not make a difference which data is stored in which subdirectory.
You can store all data in a single subdirectory or split it in several subdirectories, this will give the exact same result.

<file1>, <file2>, etc. must be comma separated CSV files without headers. For each CSV file in the `images` directory there
must exist a CSV file with an identical name in the corresponding `masks` directory. Apart from that, the names of these files
are not significant.

Each image CSV file contains the values of the pixels of an image (a pixel only contains a single value).
The corresponding mask CSV file contains the mask for the image, i.e. a matrix where each element is either 0 or 1.

Images and masks must have a resolution of exactly 512x512. Images and masks with a different resolution are ignored.
"""

import functools

import numpy as np

import utils.mask_processing
from rootdir import PROJECT_ROOT_PATH

ORIGINAL_DATASET_PATH = PROJECT_ROOT_PATH / "original dataset"
IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"


def load_original_dataset_from_disk() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Reads the original dataset from the disk. The expected format and location is described in the module documentation

    Returns :
        - A list of images, where each image is a 512x512 numpy array
        - A list of the corresponding masks, where each mask is a 512x512 numpy array containing only 0s and 1s
    """
    dataset_elements = _original_dataset_elements()

    def get_dataset_elements_with_correct_shape():
        for dataset_element in dataset_elements:
            if dataset_element.has_correct_shape():
                yield dataset_element
            else:
                print("a dataset element was ignored because its shape is not 512 x 512")

    dataset_elements_with_correct_shape = list(get_dataset_elements_with_correct_shape())

    images = [dataset_element.image_data for dataset_element in dataset_elements_with_correct_shape]
    masks = [dataset_element.mask_data for dataset_element in dataset_elements_with_correct_shape]

    clean_function = functools.partial(utils.mask_processing.remove_small_areas, max_pixel_count=4)
    masks = list(map(clean_function, masks))

    return images, masks


class _OriginalDatasetElement:
    """
    Represents an element of the original dataset.
    """

    def __init__(self, dataset_subdirectory_name: str, dataset_element_file_name: str):
        self._dataset_subdirectory_name = dataset_subdirectory_name
        self._dataset_element_file_name = dataset_element_file_name

        self.image_data = self._load_file_data(IMAGES_FOLDER_NAME)
        self.mask_data = self._load_file_data(MASKS_FOLDER_NAME)

    def _load_file_data(self, folder_name: str):
        file_path = ORIGINAL_DATASET_PATH / self._dataset_subdirectory_name / folder_name / self._dataset_element_file_name
        return np.loadtxt(file_path, delimiter=",")

    def has_correct_shape(self) -> bool:
        return self.image_data.shape == (512, 512) and self.mask_data.shape == (512, 512)


def _original_dataset_elements() -> list[_OriginalDatasetElement]:
    dataset_elements = [
        _OriginalDatasetElement(dataset_subdirectory_path.name, file_path.name)
        for dataset_subdirectory_path in ORIGINAL_DATASET_PATH.iterdir()
        for file_path in (dataset_subdirectory_path / IMAGES_FOLDER_NAME).iterdir()
    ]

    return dataset_elements
