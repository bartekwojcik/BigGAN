import tensorflow as tf


class BigGAN:
    """
    Implementation of BigGan from  https://arxiv.org/pdf/1809.11096.pdf
    """
    def __init__(self, beta_1 = 0,
                 beta_2=0.999,
                 D_lr=0.0002,
                 G_lr=0.00005):
        """Parameters come from Apendix C"""


    def define_generator(self):
        # for embeding use keras embeding layer /shrug
        '''
        https://voletiv.github.io/docs/presentations/20181030_Mila_BigGAN.pdf
        shared embedding
        '''

