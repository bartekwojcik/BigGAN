import tensorflow as tf
from tensorflow.keras.layers import Reshape
from src.operations import split_vector, orthogonal_regularizer_fully
from src.cust_layers.wrappers.fully_connected_sn import fully_connected_sn
from src.cust_layers.wrappers.resblock_up_condition import resblock_up_condition


weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)


class BigGAN128:
    """
    Implementation of BigGan from  https://arxiv.org/pdf/1809.11096.pdf
    """

    def __init__(
        self, n_channels=96, beta_1=0, beta_2=0.999, D_lr=0.0002, G_lr=0.00005
    ):
        """Parameters come from Apendix C"""

        self.n_channels = n_channels

    def define_generator(self, z_dim=120, is_trainig=True):

        use_spectral = True

        assert z_dim == 120, "for now i just want 120 noise dimention"

        in_lat = tf.keras.layers.Input(shape=(z_dim,))
        z_split = split_vector(in_lat, num_of_splits=6, axis=-1)

        ch = 16 * self.n_channels

        x = fully_connected_sn(
            z_split[0],
            units=4 * 4 * ch,
            use_spectral=use_spectral,
            kernel_regularizer=weight_regularizer_fully,
        )
        x = Reshape((4, 4, ch))(x)

        out = resblock_up_condition(x, z_split[1], channels=ch, use_bias=False, use_spectral=use_spectral)

        g_model = tf.keras.models.Model(in_lat, out)

        #delete
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        g_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        #end delete

        return g_model

        # x = resblock_up_condition(
        #     x, z_split[1], channels=ch, use_bias=False, use_spectral=use_spectral
        # )

    def OLD_define_generator(self, z_dim=120, is_trainig=True):

        in_lat = tf.keras.layers.Input(shape=(z_dim,))
        use_spectral = True

        assert z_dim == 120, "for now i just want 120 noise dimention"

        # for now i am not conditioning generator (coudnt find any good sources so first i will they way i know it works)

        z_split = tf.split(in_lat, num_or_size_splits=6, axis=-1)

        ch = 16 * self.n_channels
        x = linear(z_split[0], n_units=4 * 4 * ch)

        x = tf.reshape(x, shape=[-1, 4, 4, ch])

        # todo in these blocs include the generator path
        x = resblock_up(
            x,
            z_split[1],
            channels=ch,
            use_bias=False,
            is_training=is_trainig,
            use_spectral=use_spectral,
        )
        ch = ch // 2

        x = resblock_up(
            x,
            z_split[2],
            channels=ch,
            use_bias=False,
            is_training=is_trainig,
            use_spectral=use_spectral,
        )
        ch = ch // 2

        x = resblock_up(
            x,
            z_split[3],
            channels=ch,
            use_bias=False,
            is_training=is_trainig,
            use_spectral=use_spectral,
        )
        ch = ch // 2

        x = resblock_up(
            x,
            z_split[4],
            channels=ch,
            use_bias=False,
            is_training=is_trainig,
            use_spectral=use_spectral,
        )

        x = self_attention(x, n_channels=ch, use_spectral=use_spectral)

        ch = ch // 2

        x = resblock_up(
            x,
            z_split[5],
            channels=ch,
            use_bias=False,
            is_training=is_trainig,
            use_spectral=use_spectral,
        )

        x = tf_batch_norm(x, is_trainig)
        x = tf.nn.relu(x)
        x = conv(
            x,
            channels=3,
            kernel=3,
            stride=1,
            pad=1,
            use_bias=False,
            use_spectral=use_spectral,
        )

        out = tf.nn.tanh(x)

        g_model = tf.keras.models.Model(in_lat, out)

        return g_model

    def define_discriminator(
        self, in_shape=(128, 128, 3), n_classes=1, is_training=True
    ):

        use_spectral = True
        ch = self.n_channels

        inputs = tf.keras.layers.Input(shape=in_shape)

        a = resblock_down(
            inputs,
            channels=ch,
            use_bias=False,
            is_training=is_training,
            use_spectral=use_spectral,
        )

        b = self_attention(a, n_channels=ch, use_spectral=use_spectral)

        ch = ch * 2

        c = resblock_down(
            b,
            channels=ch,
            use_bias=False,
            is_training=is_training,
            use_spectral=use_spectral,
        )
        ch = ch * 2
        d = resblock_down(
            c,
            channels=ch,
            use_bias=False,
            is_training=is_training,
            use_spectral=use_spectral,
        )
        ch = ch * 2
        e = resblock_down(
            d,
            channels=ch,
            use_bias=False,
            is_training=is_training,
            use_spectral=use_spectral,
        )
        ch = ch * 2
        f = resblock_down(
            e,
            channels=ch,
            use_bias=False,
            is_training=is_training,
            use_spectral=use_spectral,
        )

        g = tf.nn.relu(f)

        h = global_sum_pooling(g)

        i = linear(h, n_units=n_classes, use_spectral_norm=use_spectral)

        d_model = tf.keras.models.Model(inputs, i)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        d_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return d_model

    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = tf.keras.Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        return model

    # def build_model(self):
    #
    #     self.z = tf.random.truncated_normal(shape=[self.batch_size, 1,1, self.z_dim])
    #
    #     real_logits = self.discriminator(self.inputs)
    #     fake_images = self.generator(self.z)
    #     fake_logits = self.discriminator(fake_images)
    #
    #     self.d_loss = discriminator_loss(real_logits, fake_logits)
    #     self.g_loss = generator_loss(fake_logits)
    #
    #     #how do i freeze part of the graph in Tensorflow 2?  without scopes and variable name management it is tricky
    #     #TODO meh, just rewrite it as Keras models /shrug
