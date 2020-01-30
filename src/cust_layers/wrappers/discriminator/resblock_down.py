import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Add
from src.cust_layers.wrappers.conv_block import conv_block


def reblock_down(x_init, channels, use_bias, use_spectral, kernel_regularizer):

    x = BatchNormalization(momentum=0.9, epsilon=1e-05)(x_init)
    x = ReLU()(x)
    x = conv_block(x, channels, kernel_regularizer, kernel=3, stride=2, padding='same', use_bias= use_bias, use_spectral=use_spectral)

    x = BatchNormalization(momentum=0.9, epsilon=1e-05)(x)
    x = ReLU()(x)
    x = conv_block(x, channels, kernel_regularizer, kernel=3, stride=1, padding='same', use_bias=use_bias, use_spectral=use_spectral)

    x_init = conv_block(x_init, channels, kernel_regularizer, kernel=3, stride=2, padding='same', use_spectral=use_spectral, use_bias=use_bias)

    out = Add()([x, x_init])

    return out
