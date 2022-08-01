import numpy as np
from keras import Model, Input
from keras.applications.densenet import layers
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
import keras.layers


def model_sma_detection():
    input = x = Input(shape=(512, 512))

    x = keras.layers.Reshape(target_shape=(512, 512, 1))(x)

    x = BatchNormalization()(input)

    base_channel_num = 16

    fcn1 = _fcn_layers(x, base_channel_num, 1)
    x = MaxPooling2D(pool_size=2)(fcn1)  # Downsampling
    fcn2 = _fcn_layers(x, base_channel_num * 2 ** 1, base_channel_num)
    x = MaxPooling2D(pool_size=2)(fcn2)  # Downsampling
    fcn3 = _fcn_layers(x, base_channel_num * 2 ** 2, base_channel_num * 2 ** 1)
    x = MaxPooling2D(pool_size=2)(fcn3)  # Downsampling
    fcn4 = _fcn_layers(x, base_channel_num * 2 ** 3, base_channel_num * 2 ** 2)
    x = MaxPooling2D(pool_size=2)(fcn4)  # Downsampling

    # Bottom layer
    x = Conv2D(base_channel_num * 2 ** 4, 3, activation="relu", padding="same")(x)
    # x = BatchNormalization()(x)
    x = Conv2D(base_channel_num * 2 ** 4, 3, activation="relu", padding="same")(x)
    # x = BatchNormalization()(x)

    x = _fcn_up_conv(x, fcn4, base_channel_num * 2 ** 3)
    # x = BatchNormalization()(x)
    x = _fcn_up_conv(x, fcn3, base_channel_num * 2 ** 2)
    # x = BatchNormalization()(x)
    x = _fcn_up_conv(x, fcn2, base_channel_num * 2 ** 1)
    # x = BatchNormalization()(x)
    x = _fcn_up_conv(x, fcn1, base_channel_num)
    # x = BatchNormalization()(x)

    fcn1_2 = _fcn_layers(x, base_channel_num, 1)
    x = MaxPooling2D(pool_size=2)(fcn1)  # Downsampling
    fcn2_2 = _fcn_layers(x, base_channel_num * 2 ** 1, base_channel_num)
    x = MaxPooling2D(pool_size=2)(fcn2)  # Downsampling
    fcn3_2 = _fcn_layers(x, base_channel_num * 2 ** 2, base_channel_num * 2 ** 1)
    x = MaxPooling2D(pool_size=2)(fcn3)  # Downsampling

    x = Conv2D(base_channel_num * 2 ** 3, 3, activation="relu", padding="same")(x)
    # x = BatchNormalization()(x)

    x = _fcn_up_conv(x, fcn3_2, base_channel_num * 2 ** 2)
    # x = BatchNormalization()(x)
    x = _fcn_up_conv(x, fcn2_2, base_channel_num * 2 ** 1)
    # x = BatchNormalization()(x)
    x = _fcn_up_conv(x, fcn1_2, base_channel_num)
    # x = BatchNormalization()(x)

    # Final layer
    x = Conv2D(
        1,
        1,
        activation="sigmoid",
        padding="same",
        kernel_initializer=_initializers(base_channel_num),
    )(x)

    model = Model(inputs=[input], outputs=[x])

    return model


def _initializers(num_channels):
    return RandomNormal(stddev=np.sqrt(2 / (9 * num_channels)))


def _fcn_layers(input, num_channels, num_channels_in):
    """"""
    x = Conv2D(
        num_channels,
        3,
        activation="relu",
        padding="same",
        kernel_initializer=_initializers(num_channels_in),
    )(input)
    # x = BatchNormalization()(x)
    x = Conv2D(
        num_channels,
        3,
        activation="relu",
        padding="same",
        kernel_initializer=_initializers(num_channels),
    )(x)
    return x  # BatchNormalization()(x)


def _fcn_up_conv(input_conv, input_conc, num_channels):
    """"""
    # x = Conv2DTranspose(num_channels, kernel_size=3, strides=2, padding='same', kernel_initializer=initializers(num_channels))(input_conv)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(input_conv)
    x = layers.concatenate([input_conc, x])
    # x = BatchNormalization()(x)
    x = Conv2D(
        num_channels,
        3,
        activation="relu",
        padding="same",
        kernel_initializer=_initializers(num_channels),
    )(x)
    # x = BatchNormalization()(x)
    return Conv2D(
        num_channels,
        3,
        activation="relu",
        padding="same",
        kernel_initializer=_initializers(num_channels),
    )(x)
