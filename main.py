from src.BigGAN128 import BigGAN128
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':
    gan = BigGAN128()
    z = ''

    d_model = gan.define_discriminator()
    plot_model(d_model, to_file="model.png")


