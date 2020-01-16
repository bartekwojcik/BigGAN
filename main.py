from src.BigGAN128 import BigGAN128
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':
    gan = BigGAN128()

    d_model = gan.define_discriminator()
    plot_model(d_model, to_file="discriminator_model.png")
    g_model = gan.define_generator()
    plot_model(g_model, to_file="generator_model.png")


