from src.BigGAN import BigGAN

if __name__ == '__main__':
    gan = BigGAN()
    z = ''
    gan.define_generator(z)
