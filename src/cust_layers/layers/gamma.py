import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, InputSpec, Layer


class Gamma(Layer):

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.gamma = self.add_weight(name='gamma',
                                      shape=(1),
                                      initializer=tf.keras.initializers.Constant(0.0),
                                      trainable=True)

        super(Gamma, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        o = inputs[0]
        x = inputs[1]

        out = self.gamma * o + x
        return out

    def compute_output_shape(self, input_shape):
        print("GAMMA input_shape:", input_shape)
        return input_shape