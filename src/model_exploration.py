import keras
import keras.callbacks
import keras_tuner
import keras.backend

import custom_layers
import data.preloaded.load
import model_training
import rootdir
import gc


def explore_hyper_parameters():
    tuner = keras_tuner.Hyperband(
        MyHyperModel(),
        objective=keras_tuner.Objective("val_dice", direction="max"),
        max_epochs=10,
        hyperband_iterations=1,
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


class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp: keras_tuner.HyperParameters):
        keras.backend.clear_session()
        gc.collect()
        return model_training.build_model(hp)

    def fit(self, hp: keras_tuner.HyperParameters, model: keras.Model, *args, **kwargs):
        train_validation_split_random_state = hp.Int("train val split", min_value=0, max_value=100000)
        train_dataset, validation_dataset = data.preloaded.load.train_validation_tf_datasets(random_state=train_validation_split_random_state)
        return model_training.train_model(
            hp=hp, base_model=model, train_dataset=train_dataset, validation_dataset=validation_dataset,
            *args, **kwargs)
