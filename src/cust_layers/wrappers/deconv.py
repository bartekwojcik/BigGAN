import tensorflow as tf
from src.operations import orthogonal_regularizer
from src.cust_layers.layers.conv_2d_sn_transposed import Conv_SN_2D_Transpose

weight_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
def deconv(x, channels, kernel, stride, padding='SAME', use_bias=True, use_spectral=False):
    if use_spectral:
        x = Conv_SN_2D_Transpose(filters=channels, kernel_size=kernel,
                                            kernel_initializer=weight_init,
                                            kernel_regularizer=weight_regularizer,
                                            strides=stride,
                                            padding=padding,
                                            use_bias=use_bias
                                            )(x)

    else:

        x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel,
                                            kernel_initializer=weight_init,
                                            kernel_regularizer=weight_regularizer,
                                            strides=stride,
                                            padding=padding,
                                            use_bias=use_bias
                                            )(x)

    return x