from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, concatenate


def orthogonal_regularization_ND(scale):
    """
    https://arxiv.org/pdf/1809.11096.pdf
    :param scale:
    :return:
    """
    def regularization(w):
        #_, _, _, c = w.get_shape().as_list()
        _, _, _, c = K.int_shape(w)

        w = K.reshape(w, [-1, c])

        w_t = K.transpose(w)
        identity = K.eye(c)

        w_t_w = K.dot(w_t, w)
        subtraction = w_t_w - identity

        loss = tf.nn.l2_loss(subtraction)

        return scale * loss

    return regularization

def orthogonal_regularization_2D(scale):
    """
    https://arxiv.org/pdf/1809.11096.pdf
    """
    def regularization(w):

        _,c = w.get_shape().as_list()

        w_t = tf.transpose(w)
        identity = tf.eye(c)

        w_t_w = tf.matmul(w_t, w)
        substraction = tf.subtract(w_t_w, identity)

        loss = tf.nn.l2_loss(substraction)

        return scale * loss

    return regularization

def spectral_normalisation(w, iterations=1):

    """
    taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    https://arxiv.org/pdf/1802.05957.pdf
    """

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.Variable(lambda:tf.initializers.RandomNormal(stddev=1.0)(shape=[1,w_shape[-1]]), trainable=False)

    u_hat = u
    v_hat = None

    for i in range(iterations):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w /sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def hw_flatten(x):
    #meh
    # return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])
    sh = x.get_shape().as_list()
    return tf.keras.layers.Reshape((sh[1] * sh[2], sh[3]))(x)



def discriminator_loss(real, fake):
    # hinge loss, SAGAN PAPER https://arxiv.org/pdf/1805.08318.pdf
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real)) #todo shoudnt it be -1*tf.reduce_mean(relu(-1+real))?
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake)) #todo shoudnt it be -1*tf.reduce_mean(relu(-1-fake))?

    return real_loss + fake_loss

def generator_loss(fake):

    fake_loss = -tf.reduce_mean(fake)

    return fake_loss












