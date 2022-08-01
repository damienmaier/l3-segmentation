import keras
import keras.layers
import keras_tuner

import custom_layers


def get_preprocessing_model(train_dataset, hp: keras_tuner.HyperParameters = keras_tuner.HyperParameters()):
    preprocessing_model = keras.Sequential()
    preprocessing_model.add(keras.Input(shape=(512, 512)))

    gaussian_noise_standard_deviation = hp.Float("gaussian noise", min_value=0, max_value=1, default=0.5)
    preprocessing_model.add(keras.layers.GaussianNoise(gaussian_noise_standard_deviation))

    if hp.Boolean("horizontal flip", default=True):
        preprocessing_model.add(keras.layers.RandomFlip(mode="horizontal"))

    if hp.Boolean("clip preprocessing", default=True):
        preprocessing_model.add = custom_layers.ClipLayer()

    if hp.Boolean("data normalization", default=True):
        normalization_layer = keras.layers.Normalization(axis=None)
        normalization_layer.adapt(train_dataset)
        preprocessing_model.add(normalization_layer)

    return preprocessing_model
