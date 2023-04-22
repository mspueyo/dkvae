from keras.datasets import mnist

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
    x_train, y_train, _, _ = load_mnist()
    autoencoder = train(x_train[:1000], x_train[:1000], AE_LEARNING_RATE, AE_BATCH_SIZE, AE_EPOCHS)
    autoencoder.save("model")
    autoencoder2 = VariationalAutoencoder.load("model")
    autoencoder2.model.summary()
