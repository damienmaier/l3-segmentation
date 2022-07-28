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
import gc


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
    hyper_model = MyHyperModel()
    hp = keras_tuner.HyperParameters()
    model = hyper_model.build(hp)
    model.summary()
    hyper_model.fit(hp, model)


def _add_final_round_layer(model: keras.Model) -> keras.Model:
    x = model.output
    x = arch.custom_layers.RoundLayer()(x)
    model_with_final_round_layer = keras.Model(model.input, x)
    return model_with_final_round_layer


def _create_dataset_from_data(images: np.ndarray, masks: np.ndarray, batch_size: int,
                              add_pixel_weights: bool) -> tf.data.Dataset:
    if add_pixel_weights:
        def get_pixels_weights_values_for_mask(mask: np.ndarray):
            weight_for_0, weight_for_1 = sklearn.utils.class_weight.compute_class_weight(
                class_weight="balanced",
                classes=[0, 1],
                y=mask.flat
            )
            return weight_for_0, weight_for_1

        pixel_weights_values = np.array(list(map(get_pixels_weights_values_for_mask, masks)))
        base_dataset_compressed_weights = tf.data.Dataset.from_tensor_slices((images, masks, pixel_weights_values))

        def get_image_mask_and_weights_array(image, mask, weights_values):
            weights_array_if_full_0 = tf.fill(dims=mask.shape, value=weights_values[0])
            weights_array_if_full_1 = tf.fill(dims=mask.shape, value=weights_values[1])
            weights_array = tf.where(condition=(mask == 0), x=weights_array_if_full_0, y=weights_array_if_full_1)
            weights_array = tf.reshape(weights_array, shape=(512, 512, 1))
            return image, mask, weights_array

        base_dataset = base_dataset_compressed_weights.map(get_image_mask_and_weights_array)
    else:
        base_dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    prepared_dataset = base_dataset.shuffle(len(images)).batch(batch_size).prefetch(buffer_size=1)
    return prepared_dataset


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

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model,
            *args, **kwargs):

        images_train, images_validation, masks_train, masks_validation = dataset.data_loading.get_random_train_validation_split()

        use_weighted_loss = hp.Boolean("weighted loss", default=True)

        train_dataset = _create_dataset_from_data(images_train, masks_train,
                                                  batch_size=config.TRAINING_BATCH_SIZE,
                                                  add_pixel_weights=use_weighted_loss)
        validation_dataset = _create_dataset_from_data(images_validation, masks_validation,
                                                       batch_size=config.TRAINING_BATCH_SIZE,
                                                       add_pixel_weights=use_weighted_loss)
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
