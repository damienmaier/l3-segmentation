import numpy as np

from rootdir import PROJECT_ROOT_PATH

ORIGINAL_DATASET_PATH = PROJECT_ROOT_PATH / "original dataset"
IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"


def _load_original_dataset_from_disk():
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

    return images, masks


class OriginalDatasetElement:
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


def _original_dataset_elements() -> list[OriginalDatasetElement]:
    dataset_elements = [
        OriginalDatasetElement(dataset_subdirectory_path.name, file_path.name)
        for dataset_subdirectory_path in ORIGINAL_DATASET_PATH.iterdir()
        for file_path in (dataset_subdirectory_path / IMAGES_FOLDER_NAME).iterdir()
    ]

    return dataset_elements
