from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, concatenate


def split_vector(input, num_of_splits=6, axis=-1)-> List:
    branch_outputs = []
    _, l = input.get_shape().as_list()
    length_per_block = l // 6

    for i in range(num_of_splits):

        out = Lambda(lambda x: x[:,length_per_block*i:length_per_block*(i+1)])(input)
        branch_outputs.append(out)

    return branch_outputs


def orthogonal_regularizer_fully(scale) :

    def orth(w):

        _,c = w.get_shape().as_list()
        identity = tf.eye(c)
        w_transpose = K.transpose(w)
        w_mul = K.dot(w_transpose, w)
        reg = w_mul - identity

        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return orth

def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


