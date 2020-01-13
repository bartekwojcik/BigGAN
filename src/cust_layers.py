import tensorflow as tf

from src.operations import spectral_normalisation

truncated_normal_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
orthogonal_regularizer =

def linear(x, n_units, use_bias=True, use_spectral_norm= True, scope="linear"):

    x = tf.keras.layers.Flatten()(x)
    shape = x.get_shape().as_list()
    n_channels = shape[-1]

    if use_spectral_norm:
        w = tf.Variable(name="weights", shape=[n_channels, n_units], dtype=tf.float32, initial_value=truncated_normal_init)
        bias =  tf.Variable(name="bias", shape=[n_units], initial_value=tf.keras.initializers.Constant(0.0))

        x = tf.matmul(x, spectral_normalisation(w)) + bias
        return x
    else:
        x = tf.keras.layers.Dense(units=n_units, kernel_initializer=truncated_normal_init, kernel_regularizer=orthogonal_regularizer,
                                  use_bias=use_bias)


def resblock_up(x, param, channels, use_bias, is_training, use_spectral):
    pass





