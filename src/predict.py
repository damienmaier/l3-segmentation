import keras.models
import numpy as np

from model_training import MODEL_PATH


def predict(image):
    print("begin loading model")
    model = keras.models.load_model(MODEL_PATH)
    print("end loading model")
    print("begin prediction")
    model_output = model.predict(np.array([image]))
    print("end prediction")

    model_output_2d = np.squeeze(model_output)
    predicted_mask = (model_output_2d > 0.5).astype(int)

    return predicted_mask
