import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Add
from src.cust_layers.wrappers.conv_block import conv_block

def resblock(x_init, channel, kernel_regularizer, use_bias=True, use_spectral=True):

    x = conv_block(x_init, channel, kernel_regularizer, kernel=3, stride=1, padding='same', use_bias=use_bias,use_spectral=use_spectral)
    x = BatchNormalization(momentum=0.9,epsilon=1e-05)(x)
    x = ReLU()(x)

    x = conv_block(x, channel, kernel_regularizer, kernel=3, stride=1, padding='same', use_bias=use_bias,use_spectral=use_spectral)
    x = BatchNormalization(momentum=0.9,epsilon=1e-05)(x)

    out = Add()([x, x_init])
    return out
