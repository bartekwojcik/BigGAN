from src.BigGAN128 import BigGAN128
from tensorflow.keras.utils import plot_model
from keras.datasets.cifar10 import load_data
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
import tensorflow as tf

IMG_FOLDER_PATH = r"C:\Users\barte\Desktop\Cifar10-128"

def print_model(model):
    stringlist = []
    model.summary(print_fn=stringlist.append)
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

def resize_image(image, w, h,i):
    img_path = os.path.join(IMG_FOLDER_PATH, f"img_{i}.png")
    Image.fromarray(image).resize((w,h)).save(img_path)

def resize_and_save_to_file(desired_width=128, desired_height=128):

    (trainX, trainY), (_,_) = load_data()
    sh = trainX.shape

    print('resizing')
    for i in range(sh[0]):
        img = trainX[i]
        resize_image(img, desired_width,desired_height,i)

        if i % 1000 == 0:
            print(i)

def load_real_y():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    return trainy

def get_images(path, ix):
    func = lambda f: f.split('_')[1].split('.')[0]
    images_list = os.listdir(path)
    images_list.sort(key= lambda f: int(func(f)))
    X = np.zeros((len(ix), 128,128,3), dtype='float32')
    for index,i in enumerate(ix):
        img_path = os.path.join(IMG_FOLDER_PATH,images_list[i])
        img = np.array(Image.open(img_path))
        img = img.astype('float32')
        img = (img - 127.5) / 127.5
        X[index] = img

    return X


def generate_real_samples(labels, n_samples, num_img_in_dataset=50000):
    ix = np.random.randint(0, num_img_in_dataset, n_samples)
    x, labels = get_images(IMG_FOLDER_PATH,ix), labels[ix]
    y = np.ones((n_samples,1))
    return [x,labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = np.zeros((n_samples, 1))
    return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, all_labels, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = 'd_model_%04d.h5' % (step+1)
    d_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))


def train(g_model, d_model, gan_model, all_labels, latent_dim, n_epochs=100, n_batch=16):
    tf.compat.v1.enable_eager_execution()
    tf.executing_eagerly()

    bat_per_epo = int(all_labels.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(all_labels, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(z_input, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            if True: # TODO if (i + 1) % (bat_per_epo * 1) == 0:
                summarize_performance(i, g_model, d_model, latent_dim, all_labels)
    # save the generator model
    g_model.save('cgan_generator.h5')

if __name__ == '__main__':
    latent_dim = 120
    base = BigGAN128()

    d_model = base.define_discriminator()
    plot_model(d_model, to_file="discriminator.png")
    print_model(d_model)

    g_model = base.define_generator(latent_dim)
    plot_model(g_model, to_file="generator.png")
    print_model(g_model)


    gan_model = base.define_gan(g_model,d_model)
    print(gan_model.summary())
    labels = load_real_y()
    # train model
    train(g_model, d_model, gan_model, labels, latent_dim)







