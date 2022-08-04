import data.original_dataset
import data.preloaded.load
import data.preloaded.save
import final_model
import model_evaluation
import model_exploration
import rootdir

TEST_SET_PREDICTIONS_PATH = rootdir.PROJECT_ROOT_PATH / "test set predicted masks.npy"


def create_preloaded_dataset_from_original_dataset():
    images, masks = data.original_dataset.load_original_dataset_from_disk()
    data.preloaded.save.preload_original_dataset(images, masks)


def explore_models():
    model_exploration.explore_hyper_parameters()


def train_best_model():
    final_model.train_final_model()


def compute_predictions_for_test_set():
    test_dataset = data.preloaded.load.test_tf_dataset(shuffle=False)
    predicted_masks = final_model.predict(test_dataset)
    data.preloaded.save.save_test_predictions(predicted_masks)


def evaluate_performance_of_predictions_on_test_set():
    images_dataset, true_masks_dataset, predictions_dataset = data.preloaded.load.test_images_masks_predictions_tf_datasets()
    model_evaluation.model_performance_summary(
        images=images_dataset,
        blue_masks=true_masks_dataset, red_masks=predictions_dataset,
        blue_mask_legend="true segmentation", red_mask_legend="model segmentation",
        images_display_count=10, display_box_plots=True
    )

# -------- Prepare dataset --------
# create_preloaded_dataset_from_original_dataset()

# -------- Tune model --------
# explore_models()

# -------- Train chosen model on train set --------
# train_best_model()

# -------- Compute predictions on test set --------
# compute_predictions_for_test_set()

# -------- Visualize performance --------
# evaluate_performance_of_predictions_on_test_set()
