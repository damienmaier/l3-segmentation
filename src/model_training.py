import keras
import keras.callbacks
import keras_tuner
import numpy as np
import sklearn.model_selection
import tensorflow as tf

import arch.custom_layers
import arch.model_building.joachim
import config
import dataset.data_loading
import model_evaluation
import rootdir


def explore_hyper_parameters():
    images, masks = dataset.data_loading.get_train_set()

    tuner = keras_tuner.RandomSearch(
        MyHyperModel(),
        objective=keras_tuner.Objective("val_dice_metric_for_tf_model", direction="max"),
        max_trials=3,
        directory=rootdir.PROJECT_ROOT_PATH / "model tuning",
        project_name="test"
    )

    tuner.search(images, masks,
                 callbacks=[keras.callbacks.TensorBoard(rootdir.PROJECT_ROOT_PATH / "model tuning" / "tensorboard")])


def train_default_model():
    images, masks = dataset.data_loading.get_train_set()
    hyper_model = MyHyperModel()
    hp = keras_tuner.HyperParameters()
    model = hyper_model.build(hp)
    model.summary()
    hyper_model.fit(hp, model, images, masks)


def _add_final_round_layer(model: keras.Model) -> keras.Model:
    x = model.output
    x = arch.custom_layers.RoundLayer()(x)
    model_with_final_round_layer = keras.Model(model.input, x)
    return model_with_final_round_layer


def _create_dataset_from_data(data, batch_size: int) -> tf.data.Dataset:
    print(1)
    base_dataset = tf.data.Dataset.from_tensor_slices(data)
    print(2)
    prepared_dataset = base_dataset.shuffle(len(data[0])).batch(batch_size).prefetch(buffer_size=1)
    print(3)
    return prepared_dataset


def _get_mask_pixels_weights(mask: np.ndarray):
    weight0, weight1 = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=[0, 1],
                                                                       y=mask.flat)
    weights = np.zeros_like(mask)
    weights[mask == 0] = weight0
    weights[mask == 1] = weight1
    return weights


class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp: keras_tuner.HyperParameters):
        base_model = arch.model_building.joachim.model_sma_detection()

        if hp.Boolean("clip input", default=False):
            model = keras.models.Sequential()
            model.add(keras.Input(shape=(512, 512)))
            model.add(arch.custom_layers.ClipLayer())
            model.add(base_model)
        else:
            model = base_model

        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=_dice_metric,
            # avoid tf complaining from the fact that we give pixel weights without having a weighted metric
            weighted_metrics=[]
        )

        return model

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model, images: np.ndarray, masks: np.ndarray,
            *args, **kwargs):
        print("a")
        pixels_weights = np.array(list(map(_get_mask_pixels_weights, masks)))
        print("b")
        # for an unknown reason, it is necessary for the weights to have this shape
        pixels_weights = pixels_weights.reshape(-1, 512, 512, 1)
        print("c")

        (images_train, images_validation,
         masks_train, masks_validation,
         pixels_weights_train, pixels_weights_validation) = \
            sklearn.model_selection.train_test_split(images, masks, pixels_weights)
        print("d")
        if hp.Boolean("weighted loss", default=False):
            train_data = images_train, masks_train, pixels_weights_train
            validation_data = images_validation, masks_validation, pixels_weights_validation
        else:
            train_data = images_train, masks_train
            validation_data = images_validation, masks_validation

        print("e")
        train_dataset = _create_dataset_from_data(train_data, config.TRAINING_BATCH_SIZE)
        validation_dataset = _create_dataset_from_data(validation_data, config.PREDICTION_BATCH_SIZE)

        print("f")
        print(next(iter(train_dataset)))

        return model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            *args, **kwargs
        )


def _dice_metric(true_masks: tf.Tensor, model_outputs: tf.Tensor):
    predicted_masks = arch.custom_layers.RoundLayer()(model_outputs)
    return model_evaluation.dice_coefficients_between_multiple_pairs_of_masks(true_masks, predicted_masks)
