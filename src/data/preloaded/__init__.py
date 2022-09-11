"""
Module for creating the preloaded dataset on the disk and for reading data from it.

The dataset used with this project must first be "preloaded" i.e. stored on the disk in a format appropriate
for the model training process. Here "preloaded" does not mean that the data is loaded in memory, it is
only stored on the disk in a format that will allow to efficiently load it in memory in the future.

In the preloaded dataset, each image and mask is stored as a Tensorflow tensor saved in individual files.
An image is a 2D array of values and a mask is a 2D array of 0s and 1s.

The preloaded dataset contains a training set and a test set. Each set contains images and the corresponding true masks.
The test set may also contain the masks predicted by the model for its images.

The file structure is the following :
<project root>
    preloaded dataset
        train set
            images
                00000
                00001
                00002
                ...
            masks
                00000
                00001
                00002
                ...
        test set
            images
                00000
                00001
                00002
                ...
            masks
                00000
                00001
                00002
                ...
            predictions
                00000
                00001
                00002
                ...
"""

from rootdir import PROJECT_ROOT_PATH

PRELOADED_DATASET_PATH = PROJECT_ROOT_PATH / "preloaded dataset"
TRAIN_SET_PATH = PRELOADED_DATASET_PATH / "train set"
TEST_SET_PATH = PRELOADED_DATASET_PATH / "test set"
IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"
PREDICTIONS_FOLDER_NAME = "predictions"
