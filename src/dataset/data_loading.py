import numpy as np
import sklearn.model_selection

from dataset.original_dataset import _load_original_dataset_from_disk
from rootdir import PROJECT_ROOT_PATH

PRELOADED_DATASET_PATH = PROJECT_ROOT_PATH / "preloaded dataset"
TRAIN_SET_FILE_PATH = PRELOADED_DATASET_PATH / "train set.npy"
TEST_SET_FILE_PATH = PRELOADED_DATASET_PATH / "test set.npy"


def get_train_set():
    print("begin loading train set")
    images, masks = np.load(TRAIN_SET_FILE_PATH)
    print("end loading train set")
    return images, masks


def get_test_set():
    print("begin loading test set")
    images, masks = np.load(TEST_SET_FILE_PATH)
    print("end loading test set")
    return images, masks


def preload_original_dataset():
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
    """
    if PRELOADED_DATASET_PATH.exists():
        print("Error : a preloaded dataset already exists")
    else:
        PRELOADED_DATASET_PATH.mkdir()
        images, masks = _load_original_dataset_from_disk()
        images_train, images_test, masks_train, masks_test = \
            sklearn.model_selection.train_test_split(images, masks, random_state=42)

        _save_images_and_masks_on_disk(images_train, masks_train, TRAIN_SET_FILE_PATH)
        _save_images_and_masks_on_disk(images_test, masks_test, TEST_SET_FILE_PATH)


def _save_images_and_masks_on_disk(images, masks, file_path):
    array_to_save = np.array([images, masks])
    np.save(file_path, array_to_save)
