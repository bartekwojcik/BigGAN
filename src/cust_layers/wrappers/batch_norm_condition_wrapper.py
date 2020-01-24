import tensorflow
from tensorflow.keras.layers import Dense, Reshape
from src.cust_layers.layers.batch_norm_condition import BatchNormCondition


def batch_norm_layer(x,z):

    _,_,_,c = x.get_shape().as_list()
    beta = Dense(units=c)(z)
    gamma = Dense(units=c)(z)

    beta = Reshape((1,1,c))(beta)
    gamma = Reshape((1,1,c))(gamma)

    bn_out = BatchNormCondition()([x,beta,gamma])

    return bn_out


