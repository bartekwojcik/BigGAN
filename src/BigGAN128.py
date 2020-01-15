import tensorflow as tf
from src.cust_layers import linear, resblock_up, self_attention, conv, resblock_down, tf_batch_norm, global_sum_pooling





class BigGAN128:
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

        use_spectral = True
        z_dim = z.get_shape().as_list()[3]
        assert z_dim == 120, "for now i just want 120 noise dimention"

        #for now i am not conditioning generator (coudnt find any good sources so first i will they way i know it works)

        z_split = tf.split(z, num_or_size_splits = 6, axis=-1)

        n_channels = self.n_channels

        ch = 16 * self.n_channels
        x = linear(z_split[0], n_units= 4 * 4 * ch)

        x = tf.reshape(x, shape=[-1,4,4,ch])


        #todo in these blocs include the generator path
        x = resblock_up(x, z_split[1], channels=ch, use_bias=False, is_training=is_trainig, use_spectral= use_spectral)
        ch = ch // 2

        x = resblock_up(x, z_split[2], channels=ch, use_bias=False, is_training=is_trainig, use_spectral=use_spectral)
        ch = ch // 2

        x = resblock_up(x, z_split[3], channels=ch, use_bias=False, is_training=is_trainig, use_spectral=use_spectral)
        ch = ch // 2

        x = resblock_up(x, z_split[4], channels=ch, use_bias=False, is_training=is_trainig, use_spectral=use_spectral)

        x = self_attention(x, n_channels= ch, use_spectral=use_spectral)

        ch = ch // 2

        x = resblock_up(x, z_split[5], channels=ch, use_bias=False, is_training=is_trainig, use_spectral=use_spectral)

        x = tf_batch_norm(x,is_trainig)
        x = tf.nn.relu(x)
        x = conv(x, channels= 3, kernel=3, stride=1, pad=1, use_bias=False, use_spectral=use_spectral)

        x = tf.nn.tanh(x)

        return x

    def define_discriminator(self, x, is_training=True):

        use_spectral=True
        ch = self.n_channels

        x = resblock_down(x, channels=ch, use_bias = False, is_training=is_training, use_spectral=use_spectral)

        x = self_attention(x, n_channels=ch, use_spectral=use_spectral)

        ch = ch *2

        x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, use_spectral=use_spectral)
        ch = ch * 2
        x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, use_spectral=use_spectral)
        ch = ch * 2
        x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, use_spectral=use_spectral)
        ch = ch * 2
        x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, use_spectral=use_spectral)

        x = tf.nn.relu(x)

        x = global_sum_pooling(x)

        x = linear(x, n_units=1, use_spectral_norm=use_spectral)

        return x




















