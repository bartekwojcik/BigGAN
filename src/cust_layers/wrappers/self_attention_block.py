import tensorflow as tf
from src.cust_layers.wrappers.conv_block import conv_block
from tensorflow.keras.layers import MaxPool2D, Lambda, Softmax, Reshape, Permute, Dot
import tensorflow.keras.backend as K
from src.cust_layers.wrappers.conv_block import conv_block
from src.cust_layers.layers.gamma import Gamma

def keras_dot(inputs):
    x = inputs[0]
    y = inputs[1]

    out =  K.dot(x,y)
    return out

def keras_tranpose(inputs):
    return K.transpose(inputs)

def self_attention_block(x, channels,kernel_regularizer, use_spectral=False, ):
    x_shape = x.get_shape().as_list()
    f = conv_block(x, channels // 8, kernel=1, stride=1, use_spectral=use_spectral,kernel_regularizer=kernel_regularizer)
    f = MaxPool2D(pool_size=2, strides=2, padding='same')(f)

    g = conv_block(x, channels// 8, kernel=1, stride=1, use_spectral=use_spectral,kernel_regularizer=kernel_regularizer)

    h = conv_block(x, channels// 2, kernel=1, stride=1, use_spectral=use_spectral,kernel_regularizer=kernel_regularizer)
    h = MaxPool2D(pool_size=2, strides=2, padding='same')(h)

    g_shape = g.get_shape().as_list()
    hw_flatten_g = Reshape((g_shape[1]*g_shape[2],g_shape[-1]))(g)
    f_shape = f.get_shape().as_list()
    hw_flatten_f = Reshape((f_shape[1]*f_shape[2],f_shape[-1]))(f)

    #hw_flatten_g_shape = hw_flatten_g.get_shape().as_list()
    #hw_flatten_f_shape = hw_flatten_f.get_shape().as_list()

    #hw_flatten_f_transposed_shape = (hw_flatten_g_shape[2], hw_flatten_g_shape[1])
    #hw_flatten_f_transposed = Lambda(lambda x: keras_tranpose(x), output_shape=hw_flatten_f_transposed_shape)(hw_flatten_f)
    hw_flatten_f_transposed = Permute((2,1))(hw_flatten_f)

    #out_shape = (hw_flatten_g_shape[1], hw_flatten_f_shape[1])
    #s = Lambda(lambda x: keras_dot(x), output_shape=out_shape)([hw_flatten_g, hw_flatten_f_transposed])
    s = Dot(axes=(2,1))([hw_flatten_g, hw_flatten_f_transposed])

    beta = Softmax()(s)

    h_shape = h.get_shape().as_list()
    hw_flatten_h = Reshape((h_shape[1]*h_shape[2],h_shape[-1]))(h)

    #out_shape = (beta.get_shape().as_list()[1],hw_flatten_h.get_shape().as_list()[-1] )

    #o = Lambda(lambda x: keras_dot(x), output_shape=out_shape)([beta, hw_flatten_h])
    o = Dot(axes=(2,1))([beta, hw_flatten_h])
    o = Reshape((x_shape[1], x_shape[2], channels //2 ))(o)
    o = conv_block(o,channels, kernel=1, stride=1, use_spectral=use_spectral,kernel_regularizer=kernel_regularizer)

    o_shape = o.get_shape().as_list()
    out = Gamma(output_shape=(o_shape[1], o_shape[-1]))([o, x])

    return out




