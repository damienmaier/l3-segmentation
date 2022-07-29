import keras
import keras.callbacks
import keras_tuner

import custom_layers
import dataset.data_loading
import model_training
import rootdir


def explore_hyper_parameters():
    tuner = keras_tuner.RandomSearch(
        MyHyperModel(),
        objective=keras_tuner.Objective("val_dice_metric_for_tf_model", direction="max"),
        max_trials=3,
        directory=rootdir.PROJECT_ROOT_PATH / "model tuning",
        project_name="test"
    )

    tuner.search(callbacks=[keras.callbacks.TensorBoard(rootdir.PROJECT_ROOT_PATH / "model tuning" / "tensorboard")])


def train_default_model():
    hyper_model = MyHyperModel()
    hp = keras_tuner.HyperParameters()
    model = hyper_model.build(hp)
    model.summary()
    hyper_model.fit(hp, model)


def _add_final_round_layer(model: keras.Model) -> keras.Model:
    x = model.output
    x = custom_layers.RoundLayer()(x)
    model_with_final_round_layer = keras.Model(model.input, x)
    return model_with_final_round_layer


class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp: keras_tuner.HyperParameters):
        return model_training.build_model(hp)

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model, *args, **kwargs):
        images_paths, masks_paths = dataset.data_loading.get_train_set()
        return model_training.train_model(
            hp=hp, model=model, images_paths=images_paths, masks_paths=masks_paths, use_validation_set=True,
            *args, **kwargs)
