import keras
import tensorflow as tf
import keras.layers
import keras_tuner

import config
import custom_keras_objects
import data.preloaded.load
import rootdir
from model_training import build_model, train_model

MODEL_PATH = rootdir.PROJECT_ROOT_PATH / "model"


def train_final_model():
    hp = keras_tuner.HyperParameters()
    model = build_model(hp)
    train_dataset = data.preloaded.load.train_tf_dataset(shuffle=True)
    train_model(hp, model=model, train_dataset=train_dataset)

    final_model = keras.Sequential()
    final_model.add(keras.Input(shape=(512, 512)))
    final_model.add(keras.layers.Reshape(target_shape=(512, 512, 1)))
    final_model.add(model)
    final_model.add(post_processing_model)
    final_model.compile()

    final_model.save(MODEL_PATH)


def _build_post_processing_model():
    post_processing_model_ = keras.Sequential()
    post_processing_model_.add(keras.Input(shape=(512, 512, 1)))
    post_processing_model_.add(custom_keras_objects.RoundLayer())
    post_processing_model_.add(keras.layers.Reshape(target_shape=(512, 512)))
    return post_processing_model_


post_processing_model = _build_post_processing_model()


def predict(images: tf.data.Dataset) -> tf.Tensor:
    batched_images = images.batch(config.PREDICTION_BATCH_SIZE)
    model = keras.models.load_model(MODEL_PATH)
    predicted_masks = model.predict(batched_images)
    return predicted_masks
