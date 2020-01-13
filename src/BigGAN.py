import tensorflow as tf
from src.cust_layers import linear





class BigGAN:
    """
    Implementation of BigGan from  https://arxiv.org/pdf/1809.11096.pdf
    """
    def __init__(self, n_channels = 96 , beta_1 = 0,
                 beta_2=0.999,
                 D_lr=0.0002,
                 G_lr=0.00005):
        """Parameters come from Apendix C"""

        self.n_channels = n_channels


    def define_generator(self, z, is_trainig=True):

        z_dim = z.get_shape().as_list()[3]
        assert z_dim == 120, "for now i just want 120 noise dimention"

        #for now i am not conditioning generator (coudnt find any good sources so first i will they way i know it works)

        z_split = tf.split(z, num_or_size_splits = 6, axis=-1)

        n_channels = self.n_channels

        ch = 16 * self.n_channels
        x = linear(z_split[0], n_units= 4 * 4 * ch)

        x = tf.reshape(x, shape=[-1,4,4,ch])

        x = resblock_up(x, z_split(1), channels=ch, use_bias=False, is_training=is_trainig, use_spectral= True)








