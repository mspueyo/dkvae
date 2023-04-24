from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from config import *
from src.models.autoencoder import Autoencoder
from src.models.variationalautoencoder import VariationalAutoencoder


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, x_test, y_test


def train(x_train, y_train, learning_rate, batch_size, epochs):
    autoencoder = VariationalAutoencoder(
        input_shape = AE_INPUT_SHAPE,
        filters = AE_CONV_FILTERS,
        kernels = AE_CONV_KERNELS,
        strides = AE_CONV_STRIDES,
        latent_space_shape = AE_LATENT_SPACE_SHAPE
        )
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, batch_size, epochs)
    return autoencoder



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    autoencoder = train(x_train, x_train, AE_LEARNING_RATE, AE_BATCH_SIZE, AE_EPOCHS)
    autoencoder.save("model")
    autoencoder = VariationalAutoencoder.load("model")
    autoencoder.model.summary()
    x_decoded = autoencoder.model.predict(x_test)

    latent_vectors = autoencoder.encoder.predict(x_test)
    for i, latent_vector in enumerate(latent_vectors):
        print(f"Latent space vector {i + 1}: {latent_vector}")

    # Plot
    n = 10
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    # Generate a 4x4 grid of latent space points
    n = 4
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # z_sample = np.array([[xi, yi]])

            # z_sample = np.random.normal(0.5, 0.333, size=(1, latent_space_shape))
            z_sample = np.random.uniform(0, 1, size=(1, latent_space_shape))
            # z_sample = np.random.uniform(-10, 10, size=(1, latent_space_shape))

            # Clip to 0, 1
            z_sample = np.clip(z_sample, 0, 1)

            x_decoded = autoencoder.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    # start_range = digit_size // 2
    # end_range = n * digit_size + start_range + 1
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


