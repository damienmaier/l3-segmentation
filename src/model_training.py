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


def _create_fit_dataset(images_paths: np.ndarray, masks_paths: np.ndarray,
                        batch_size: int, add_pixel_weights: bool) -> tf.data.Dataset:
    images_dataset = dataset.data_loading.get_tf_dataset_from_tensor_file_paths(images_paths)
    masks_dataset = dataset.data_loading.get_tf_dataset_from_tensor_file_paths(masks_paths)

    base_dataset = tf.data.Dataset.zip(datasets=(images_dataset, masks_dataset))

    if add_pixel_weights:
        def add_pixel_weights_to_dataset_element(image, mask):
            mask_0_count = tf.cast(tf.math.count_nonzero(mask == 1), tf.float64)
            mask_1_count = tf.cast(tf.math.count_nonzero(mask), tf.float64)
            mask_size = tf.cast(tf.size(mask), tf.float64)

            weight_for_0 = mask_size / 2. / mask_0_count
            weight_for_1 = mask_size / 2. / mask_1_count
            weights = tf.stack([weight_for_0, weight_for_1])

            pixel_weights = tf.gather(weights, indices=tf.cast(mask, tf.int32))

            # for an unknown reason the weights must have this shape,
            # even if the images and masks have shape (512, 512)
            pixel_weights = tf.reshape(pixel_weights, shape=(512, 512, 1))

            return image, mask, pixel_weights

        base_dataset = base_dataset.map(add_pixel_weights_to_dataset_element)

    prepared_dataset = base_dataset.shuffle(len(images_paths)).batch(batch_size).prefetch(buffer_size=1)
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

        def dice(true_masks: tf.Tensor, model_outputs: tf.Tensor):
            predicted_masks = arch.custom_layers.RoundLayer()(model_outputs)
            return model_evaluation.dice_coefficients_between_multiple_pairs_of_masks(true_masks, predicted_masks)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=dice,
            # avoid tf complaining from the fact that we give pixel weights without having a weighted metric
            weighted_metrics=[]
        )

        return model

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model,
            *args, **kwargs):

        images_paths, masks_paths = dataset.data_loading.get_train_set()

        images_paths_train, images_paths_validation, masks_paths_train, masks_paths_validation = \
            sklearn.model_selection.train_test_split(images_paths, masks_paths)

        use_weighted_loss = hp.Boolean("weighted loss", default=False)

        train_dataset = _create_fit_dataset(images_paths_train, masks_paths_train,
                                            batch_size=config.TRAINING_BATCH_SIZE,
                                            add_pixel_weights=use_weighted_loss)
        validation_dataset = _create_fit_dataset(images_paths_validation, masks_paths_validation,
                                                 batch_size=config.TRAINING_BATCH_SIZE,
                                                 add_pixel_weights=use_weighted_loss)
        print(next(iter(train_dataset)))

        return model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            epochs=2,
            *args, **kwargs
        )
