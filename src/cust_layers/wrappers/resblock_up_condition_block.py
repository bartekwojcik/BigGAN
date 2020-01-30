from tensorflow.keras.layers import ReLU, Add

from src.operations import orthogonal_regularizer
from src.cust_layers.wrappers.batch_norm_condition_block import cond_batch_norm_block
from src.cust_layers.wrappers.deconv_block import deconv_block




def resblock_up_condition_block(x_init, z, channels,kernel_regularizer, use_bias=True, use_spectral=False):
    x = cond_batch_norm_block(x_init, z)
    x = ReLU()(x)
    x = deconv_block(
        x,
        channels,
        kernel_regularizer=kernel_regularizer,
        kernel=3,
        stride=2,
        use_bias=use_bias,
        use_spectral=use_spectral,
    )  # https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py

    x = cond_batch_norm_block(x, z)
    x = ReLU()(x)
    x = deconv_block(
        x,
        channels,
        kernel=3,
        stride=1,
        use_bias=use_bias,
        use_spectral=use_spectral,
        kernel_regularizer=kernel_regularizer,
    )

    x_init = deconv_block(
        x_init,
        channels,
        kernel=3,
        stride=2,
        use_bias=use_bias,
        use_spectral=use_spectral,
        kernel_regularizer=kernel_regularizer,
    )

    out = Add()([x, x_init])

    return out
