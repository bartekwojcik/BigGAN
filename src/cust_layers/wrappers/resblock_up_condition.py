from tensorflow.keras.layers import ReLU
from src.cust_layers.wrappers.batch_norm_condition_wrapper import batch_norm_layer
from src.cust_layers.wrappers.deconv import deconv


def resblock_up_condition(x_init, z, channels, use_bias=True, use_spectral=False):
    x = batch_norm_layer(x_init,z)
    x = ReLU()(x)
    x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, use_spectral=use_spectral) #https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py