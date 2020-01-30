import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from src.cust_layers.layers.conv_2d_sn import ConvSN2D
from src.operations import orthogonal_regularizer

weight_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)

def conv_block(x, channels,kernel_regularizer, kernel=4, stride=2, use_bias=True, use_spectral=False, padding='valid' ):

    if use_spectral:
        x = ConvSN2D(filters=channels, kernel_size=kernel, kernel_initializer=weight_init,
                   kernel_regularizer=weight_regularizer,
                     padding=padding,
                   strides=stride, use_bias=use_bias)(x)

    else:
        x = Conv2D(filters=channels,
                   kernel_size=kernel,
                   padding=padding,
                   kernel_initializer=weight_init,
                   kernel_regularizer=kernel_regularizer,
                   strides=stride, use_bias=use_bias)(x)

    return x