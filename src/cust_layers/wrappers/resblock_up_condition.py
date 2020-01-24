from tensorflow.keras.layers import ReLU
from src.cust_layers.wrappers.batch_norm_condition_wrapper import batch_norm_layer


def resblock_up_condition(x_init, z, channels, use_bias=True, use_spectral=False):
    x = batch_norm_layer(x_init,z)
    x = ReLU()(x)
    x = deconv #https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py