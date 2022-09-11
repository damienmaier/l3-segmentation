"""
This is the file to execute to run the code of this project.

At the bottom of this file there is a commented line of code for each step of this project. To run a step, juste
uncomment the corresponding line and run this file.

Each step will save its result on the disk in the location and format expected by the subsequent steps.
You can therefore run one step at a time.
"""
import data.original_dataset
import data.preloaded.load
import data.preloaded.save
import final_model
import model_evaluation
import model_exploration


def create_preloaded_dataset_from_original_dataset():
    """
    Converts the original dataset to a "preloaded" dataset, i.e. a dataset stored on the disk in a format
    appropriate for model training. The dataset is split into a training set and a test set.

    The original dataset is read on the disk from <project root> / original dataset.
    The preloaded dataset is stored on the disk at <project root> / preloaded dataset.

    The expected format for the original dataset is described in the documentation of `data.original_dataset`
    """
    images, masks = data.original_dataset.load_original_dataset_from_disk()
    data.preloaded.save.create_preloaded_dataset(images, masks)


def explore_models():
    """
    Tests several model architectures and hyperparameter values and evaluate their performances only using data
    from the training set. The data generated by this step can be visualized using TensorBoard.

    See the documentation of the `model_exploration` and `model_training` for explanation on how to control
    hyperparameter value ranges.
    """
    model_exploration.explore_hyper_parameters()


def train_final_model():
    """
    Trains the final model on the training set and saves the result on the disk.

    This function must be called after having determined the best hyperparameters for the problem.
    See the documentation of the `model_training` module for explanation on how to control the hyperparameter values
    used for the training of the final model.
    """
    final_model.train_final_model()


def compute_predictions_for_test_set():
    """
    Computes the masks predicted by the final model on the test set, and stores the result on the disk.
    """
    test_dataset = data.preloaded.load.test_tf_dataset(shuffle=False)
    predicted_masks = final_model.predict(test_dataset)
    data.preloaded.save.save_test_predictions(predicted_masks)


def evaluate_performance_of_predictions_on_test_set():
    """
    Evaluates the quality of the predictions that the final model made on the test set.

    Computes the dice coefficients and the Hausdorff distances between the predicted masks and the true masks.
    Displays images that allow to visually compare the predicted masks with the true masks.
    """
    images_dataset, true_masks_dataset, predictions_dataset = data.preloaded.load.test_images_masks_predictions_tf_datasets()
    model_evaluation.model_performance_summary(
        images=images_dataset,
        blue_masks=true_masks_dataset, red_masks=predictions_dataset,
        blue_mask_legend="true segmentation", red_mask_legend="model segmentation",
        display_image=True, display_blue_mask=True,
        images_display_count=1, display_box_plots=True
    )

# -------- Prepare the dataset --------
# create_preloaded_dataset_from_original_dataset()

# -------- Explore hyper-parameters --------
# explore_models()

# -------- Train the final model on the train set --------
# train_best_model()

# -------- Compute the predictions on the test set --------
# compute_predictions_for_test_set()

# -------- Visualize the performance --------
# evaluate_performance_of_predictions_on_test_set()
