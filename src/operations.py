import tensorflow as tf


def

def spectral_normalisation(w, iterations=1):
    "taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py"

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.Variable(name="u", shape=[1,w_shape[-1]], initial_value=tf.initializers.RandomNormal(stddev=1.0), trainable=False)

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