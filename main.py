from src.BigGAN128 import BigGAN128

if __name__ == '__main__':
    gan = BigGAN128()
    z = ''
    gan.define_generator(z)
