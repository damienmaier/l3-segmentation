"""
Module to perform model hyperparameter exploration.

The functions in this module allow to train several models with different hyperparameter values and to
visualize the impact of the hyperparameter on the model performance.

Each model is trained on a subset of the training data, the remaining part of the training data is used as a validation
set to evaluate the performance. The test set is never used.

The hyperparameters, their range and their default values are defined by the functions of the `model_training` module.
You will probably want to experiment with different hyperparameter value ranges, which you can do by modifying the
hyperparameter value ranges in the `model_training` module.

Once you have decided which hyperparameter values are optimal, you must set them as default in the `model_training`
module code and use the `final_model` module to train the final model with the default hyperparameter values
on the whole training set.
"""

import gc

import keras
import keras.backend
import keras.callbacks
import keras_tuner

import data.preloaded.load
import model_training
import rootdir


def explore_hyper_parameters() -> None:
    """
    Trains several models with different hyperparameter values and stores their performances.

    For each model, a random set of hyperparameter values is selected.
    A validation set is randomly sampled from the training set, the model is trained on the remaining data
    and its performance is evaluated on the validation set.

    The performances of the models is stored in <project root> / model tuning. This data can be analyzed with
    TensorBoard, which allows to visualize the relationship between the hyperparameter values and the
    performance of the model.
    """
    tuner = keras_tuner.RandomSearch(
        MyHyperModel(),
        objective=keras_tuner.Objective("val_dice", direction="max"),
        max_trials=10000,
        directory=rootdir.PROJECT_ROOT_PATH / "model tuning",
        project_name="test"
    )

    tuner.search(callbacks=[keras.callbacks.TensorBoard(rootdir.PROJECT_ROOT_PATH / "model tuning" / "tensorboard")])


def train_default_model() -> None:
    """
    Trains the model with the default hyperparameter values.

    This function can be useful for debugging the model training code with specific hyperparameter values.
    """
    hyper_model = MyHyperModel()
    hp = keras_tuner.HyperParameters()
    model = hyper_model.build(hp)
    hyper_model.fit(hp, model)


class MyHyperModel(keras_tuner.HyperModel):
    """
    Class that provides methods for building and training the model.

    The keras tuner in the function `explore_hyper_parameters` uses an instance of this class to build and
    train each model.
    """
    def build(self, hp: keras_tuner.HyperParameters) -> keras.Model:
        """
        Builds a tf model that must be trained using the `fit` method of this class.

        `hp` is used to control the hyperparameter values.
        """

        # I added these lines to try to reduce the amount of memory used. I am not sure that it has any effect.
        keras.backend.clear_session()
        gc.collect()

        return model_training.build_model(hp)

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model, *args, **kwargs) -> keras.callbacks.History:
        """
        Trains a model that has been previously built using the `build` method of this class.

        `hp` is used to control the hyperparameter values.

        The extra arguments are passed to the `fit` method of the model. This allows the keras tuner
        that calls the present method to provide callbacks for the training process.

        The model is trained on a subset of the training data, the remaining part of the training data is used as a validation
        set to evaluate the performance.

        Returns the history object returned by the `fit` method of the model.
        """

        # We define an hyperparameter for the validation / training split seed. This guarantees to have the same
        # training and validation sets if this function is called several times with the same hyperparameter values.
        # This is important for example when using the Hyperband tuner, because the tuner will call this function at
        # each training step of a given model, and the model must be trained on the same data during each training
        # step.
        train_validation_split_random_state = hp.Int("train val split", min_value=0, max_value=100000)
        train_dataset, validation_dataset = data.preloaded.load.train_validation_tf_datasets(
            random_state=train_validation_split_random_state)
        return model_training.train_model(
            hp=hp, model=model, train_dataset=train_dataset, validation_dataset=validation_dataset,
            *args, **kwargs)
