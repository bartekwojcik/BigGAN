import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from src.cust_layers.layers.dense_spectral_norm import DenseSpectralNorm


weight_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


def fully_connected_sn_block(inputs, units, use_spectral, kernel_regularizer, use_bias=True):
    x = Flatten()(inputs)

    if use_spectral:
        x = DenseSpectralNorm(
            units=units,
            kernel_initializer=weight_init,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
        )(x)

    else:

        x = Dense(
            units=units,
            kernel_initializer=weight_init,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
        )(x)

    return x
