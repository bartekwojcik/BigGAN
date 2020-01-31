import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer, InputSpec, BatchNormalization
from tensorflow_core.python.keras.utils import tf_utils


class BatchNormCondition(BatchNormalization):
    def build(self, input_shape):

        c = input_shape[0][-1] #there are three inputs

        self.decay = 0.9
        self.test_mean = self.add_weight(
            shape=(c),
            name="test_mean",
            dtype="float32",
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        )

        self.test_var = self.add_weight(
            shape=(c),
            name="test_mean",
            dtype="float32",
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
        )

        self.input_spec = InputSpec(shape=input_shape)
        self.built = True

    def call(self, inputs, training=None):
        x = inputs[0]
        beta = inputs[1]
        gamma = inputs[0]

        trainig_val = tf_utils.constant_value(training)
        if trainig_val == True:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2])

            #ema_mean = K.set_value(self.test_mean, self.test_mean * self.decay + batch_mean * (1 - self.decay))
            #ema_var = K.set_value(self.test_var, self.test_var * self.decay + batch_var * (1 - self.decay))

            ema_mean = tf.compat.v1.assign(self.test_mean, self.test_mean * self.decay + batch_mean * (1 - self.decay))
            ema_var = tf.compat.v1.assign(self.test_var, self.test_var * self.decay + batch_var * (1 - self.decay))

            with tf.control_dependencies([ema_mean, ema_var]):

                output = tf.nn.batch_normalization(x,batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            output = tf.nn.batch_normalization(x,self.test_mean, self.test_var, beta, gamma, self.epsilon)


        return output

    def compute_output_shape(self, input_shape):



        return input_shape
