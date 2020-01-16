import tensorflow as tf

from src.operations import spectral_normalisation, orthogonal_regularization_2D, orthogonal_regularization_ND, hw_flatten

truncated_normal_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
ortho_regr_2D = orthogonal_regularization_2D(0.0001)
ortho_regr_ND = orthogonal_regularization_ND(0.0001)


def linear(x, n_units, use_bias=True, use_spectral_norm= True, kernel_regularizer=ortho_regr_2D):

    x = tf.keras.layers.Flatten()(x)
    shape = x.get_shape().as_list()
    n_channels = shape[-1]

    if use_spectral_norm:
        w = tf.Variable(name="weights", dtype=tf.float32, initial_value=truncated_normal_init(shape=[n_channels, n_units]), constraint=kernel_regularizer)
        bias =  tf.Variable(name="bias", initial_value=tf.keras.initializers.Constant(0.0)(shape=[n_units]))

        x = tf.matmul(x, spectral_normalisation(w)) + bias
        return x

    else:
        x = tf.keras.layers.Dense(units=n_units,
                                  kernel_initializer=truncated_normal_init,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=use_bias)(x)

        return x


def conditioned_batch_norm(x, z, is_training):
    """
    https://arxiv.org/pdf/1610.07629.pdf
    :param x_init:
    :param z:
    :param is_training:
    :return:
    """

    _, _, _, c = x.get_shape().as_list()
    decay = 0.9
    epsilon = 1e-05

    test_mean = tf.Variable(dtype=tf.float32, initial_value=tf.constant_initializer(0.0)(shape=[c]), trainable=False)
    test_var = tf.Variable(dtype=tf.float32, initial_value=tf.constant_initializer(1.0)(shape=[c]), trainable=False)

    beta = linear(z,n_units=c,kernel_regularizer=None)
    gamma = linear(z,n_units=c,kernel_regularizer=None)

    beta = tf.reshape(beta, shape=[-1, 1, 1, c])
    gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

    if is_training:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

        ema_mean = tf.compat.v1.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
        ema_var = tf.compat.v1.assign(test_var, test_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([ema_mean, ema_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

    else:
        return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


def deconv(x, channels, kernel, strides, use_bias, use_spectral):

    x_shape = x.get_shape().as_list()

    output_shape = (x_shape[0], x_shape[1] * strides, x_shape[2] * strides, channels)

    if use_spectral:
        # w = tf.Variable(initial_value=truncated_normal_init(shape=[kernel, kernel, channels, x_shape[-1]]),constraint=ortho_regr_ND)
        # x = tf.nn.conv2d_transpose(x, filters=spectral_normalisation(w), output_shape=output_shape, strides=[1, strides, strides,1], padding='same' )
        #
        # if use_bias:
        #     bias = tf.Variable(shape=[channels], initial_value=tf.constant_initializer(0.0))
        #     x = tf.nn.bias_add(x, bias)


        x = tf.keras.layers.Conv2DTranspose(filters=channels,
                                            kernel_size= [kernel, kernel],
                                            kernel_initializer= truncated_normal_init,
                                            kernel_regularizer=spectral_normalisation,
                                            kernel_constraint= ortho_regr_ND,
                                            strides=[strides,strides],
                                            padding='same',
                                            use_bias=use_bias)(x)

        return x
    else:
        x = tf.keras.layers.Conv2DTranspose(filters = channels,
                                            kernel_size=kernel,
                                            kernel_initializer=truncated_normal_init,
                                            kernel_regularizer=ortho_regr_ND, padding='same')(x)

        return x


def resblock_down(x_init, channels, use_bias=True, is_training=True, use_spectral=True):

    x = tf_batch_norm(x_init, is_training)
    x = tf.keras.layers.ReLU()(x) #x = tf.nn.relu(x)
    x = conv(x, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, use_spectral=use_spectral, kernel_regularizer=None)

    x = tf_batch_norm(x, is_training)
    x = tf.nn.relu(x)

    x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, use_spectral=use_spectral, kernel_regularizer=None)

    x_init = conv(x_init, channels, kernel=3, stride=2, pad=1, use_bias=use_bias, use_spectral=use_spectral, kernel_regularizer=None)

    return x + x_init


def resblock_up(x_init, z, channels, use_bias=True, is_training=True, use_spectral=True):
    """
    https://arxiv.org/pdf/1809.11096.pdf
    :param x_init:
    :param z:
    :param channels:
    :param use_bias:
    :param is_training:
    :param use_spectral:
    :return:
    """
    x = conditioned_batch_norm(x_init, z, is_training)
    x = tf.nn.relu(x)
    x = deconv(x, channels, kernel=3, strides=2, use_bias=use_bias, use_spectral=use_spectral)

    x = conditioned_batch_norm(x, z, is_training)
    x = tf.nn.relu(x)
    x = deconv(x, channels, kernel=3, strides=1, use_bias=use_bias, use_spectral=use_spectral)

    x_init = deconv(x_init, channels, kernel=3, strides=2, use_bias=use_bias, use_spectral=use_spectral)

    return x + x_init


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, use_spectral=False,kernel_regularizer = ortho_regr_ND):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    if use_spectral:
        # w = tf.Variable(shape=[kernel, kernel, x.get_shape()[-1], channels], initial_value=truncated_normal_init, constraint=kernel_regularizer)
        #
        # x = tf.nn.conv2d(input=x, filter=spectral_normalisation(w), strides=[1,stride,stride,1], padding='valid')

        # if use_bias:
        #     bias = tf.Variable(shape=[channels], initial_value=tf.constant_initializer(0.0))
        #     x = tf.nn.bias_add(x, bias)

        x = tf.keras.layers.Conv2D(
            filters= channels,
            kernel_size= [kernel, kernel],
            kernel_initializer= truncated_normal_init,
            kernel_regularizer=spectral_normalisation,
            kernel_constraint=kernel_regularizer,
            strides=[stride,stride],
            padding='valid',
            use_bias=use_bias)(x)



    else:
        x = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel, kernel_initializer=truncated_normal_init,
                                   kernel_regularizer=kernel_regularizer, strides=stride, use_bias=use_bias)(x)

    return x


def self_attention(x, n_channels, use_spectral, ):
    """
    https://arxiv.org/pdf/1711.07971.pdf
    https://arxiv.org/pdf/1805.08318.pdf
    :param x:
    :param param:
    :param n_channels:
    :param use_bias:
    :param is_trainig:
    :param use_spectral:
    :return:
    """
    # divided by 8 because in paper C_hat = C /8
    f = conv(x, n_channels // 8, kernel=1, stride=1, use_spectral=use_spectral)
    f = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(f)

    g = conv(x, n_channels // 8, kernel=1, stride=1, use_spectral=use_spectral)

    h= conv(x, n_channels // 2, kernel=1, stride=1, use_spectral=use_spectral)
    h = tf.keras.layers.MaxPool2D(pool_size=2,strides=2, padding='SAME')(h)

    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

    beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    gamma = tf.Variable(0.0)

    sh = x.get_shape().as_list()
    o = tf.keras.layers.Reshape((sh[1], sh[2], n_channels // 2))(o)  # [bs, h, w, C]

    o = conv(o, n_channels, kernel=1, stride=1, use_spectral=use_spectral)
    x = gamma * o + x

    return x

def global_sum_pooling(x):
    return tf.reduce_sum(x, axis=[1,2])

def tf_batch_norm(x_init, is_training):
    return tf.keras.layers.BatchNormalization(momentum=0.9,
                                              epsilon=1e-05)(x_init, is_training)


























