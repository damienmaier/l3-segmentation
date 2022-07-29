import numpy as np
from tensorflow import keras

import dataset.data_loading
import dataset.original_dataset
import model_evaluation
import model_training
import predict
import rootdir

MODEL_PATH = rootdir.PROJECT_ROOT_PATH / "model"
TEST_SET_PREDICTIONS_PATH = rootdir.PROJECT_ROOT_PATH / "test set predicted masks.npy"


def create_preloaded_dataset_from_original_dataset():
    images, masks = dataset.original_dataset.load_original_dataset_from_disk()
    dataset.data_loading.preload_original_dataset(images, masks)


def find_best_model():
    train_images, train_masks = dataset.data_loading.get_train_set()
    model_training.explore_hyper_parameters(train_images, train_masks)


def compute_predictions_for_test_set():
    if TEST_SET_PREDICTIONS_PATH.exists():
        print("Error, the predictions have already been computed")
    else:
        images_test, _ = dataset.data_loading.get_test_set()
        model = keras.models.load_model(MODEL_PATH)
        predicted_masks = predict.predict(images=images_test, model=model)
        array_to_save = np.array(predicted_masks)
        np.save(TEST_SET_PREDICTIONS_PATH, array_to_save)


def evaluate_performance_of_predictions_on_test_set():
    images, true_masks = dataset.data_loading.get_test_set()
    predicted_masks = np.load(TEST_SET_PREDICTIONS_PATH)
    model_evaluation.model_performance_summary(images=images, true_masks=true_masks, predicted_masks=predicted_masks)

# -------- Prepare dataset --------
# create_preloaded_dataset_from_original_dataset()

# -------- Tune model --------
# find_best_model()

# -------- Compute predictions on test set --------
# compute_predictions_for_test_set()

# -------- Visualize performance --------
# evaluate_performance_of_predictions_on_test_set()
