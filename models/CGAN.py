"""2D Deep Convolutional GAN"""

import keras
from keras import layers
import tensorflow as tf
import numpy as np

from keras.optimizers import Adam


class DCGAN:
    def __init__(self, bank):
        in_channels = 10
        out_channels = 1000
        
        print(bank.element_spec)
        self.X_train = bank.from_tensor_slices([0])
        self.y_train = bank.from_tensor_slices([1])
        print(tf.shape(self.X_train))
        print(tf.shape(self.y_train))
        
        self.build_generator(in_channels) 
        self.build_discriminator(out_channels)
        self.define_gan()
    
    
    def build_generator(self, generator_in_channels):
        self.generator = keras.Sequential([
            keras.layers.InputLayer((generator_in_channels,)),
            layers.Dense(7 * 7 * generator_in_channels),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, generator_in_channels)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ], name="generator")
        
    def build_discriminator(self, discriminator_in_channels):
        self.discriminator = keras.Sequential([
            keras.layers.InputLayer((28, 28, discriminator_in_channels)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1)
        ], name="discriminator")

    def define_gan(self):
        self.discriminator.trainable = False
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.model = model
    