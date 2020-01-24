import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, InputSpec

from src.cust_layers.layers.ops import power_iteration


class DenseSpectralNorm(Dense):
    """
    https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py
    """

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        else:
            self.bias = None

        self.u = self.add_weight(
            shape=(1, self.kernel.shape.as_list()[-1]),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        W_shape = self.kernel.shape.as_list()
        # flatten
        W_reshaped = K.reshape( self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)

        #calculate sigma
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # normalize it

        w_bar = W_reshaped / sigma

        trainig_val = tf_utils.constant_value(training)
        if trainig_val == False:
            w_bar = K.reshape(w_bar, W_shape)

        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                w_bar = K.reshape(w_bar, W_shape)

        output = K.dot(inputs, w_bar)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.activation is not None:
            output = self.activation(output)

        print("DENSE: ", output.shape)
        return output

    def compute_output_shape(self, input_shape):
         return (input_shape[0], self.units)






















