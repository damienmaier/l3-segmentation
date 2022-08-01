# comes from https://keras.io/examples/vision/deeplabv3_plus/
import keras.layers
from keras.applications.densenet import layers
import tensorflow as tf

import custom_layers


def deep_lab_v3_plus():
    image_size = 512
    num_classes = 2

    model_input = x = tf.keras.Input(shape=(image_size, image_size, 1))

    x = custom_layers.GrayscaleToRGBLayer()(x)

    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=x
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = _DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = _convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = _convolution_block(x)
    x = _convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    x = layers.Softmax(axis=3)(x)
    x = x[:, :, :, :1]
    return tf.keras.Model(inputs=model_input, outputs=x)


def _convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    # x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def _DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = _convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = _convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = _convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = _convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = _convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = _convolution_block(x, kernel_size=1)
    return output
