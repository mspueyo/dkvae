# DK·VAE

DK·VAE (Drumkit Variational Autoencoder) is an audio generation tool based on variational autoencoders. It has been developed as a final thesis for the Data Science master's degree at UOC (Universitat Oberta de Catalunya).

The DKVAE project uses a Variational Autoencoder (VAE), a type of generative model, to learn a compact, continuous representation of drum sounds. The model is trained on a large dataset of drum sounds, and it can generate novel drum sounds that are similar to the training data. DrumKitVAE can be seen as a drum machine that has the ability to generate an infinite number of unique drum sounds, adding creativity and versatility to your music production process.


# Table of Contents

- [DK·VAE](#dk-vae)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  * [Requirements](#requirements)
  * [Notebook](#notebook)
  * [Installation](#installation-1)
- [Implementation](#implementation)
  * [Project Structure](#project-structure)
  * [DCGAN Architecture](#dcgan-architecture)
  * [VAE Architecture](#vae-architecture)
- [Dataset](#dataset)
- [Results](#results)
  * [Performance](#performance)

# Installation

## Requirements
DKVAE uses 

## Notebook
You can try DKVAE online with no installation on the following Google Colab notebook.

## Installation

Alternatively, the package can be installed locally by running the following comand:

# Implementation

## Project Structure

The project consists on a flexible and scalable architecture that includes audio preprocessing techniques and sound bank information in objects that are able to interact within them. The most important objects that are included are:

* config.py
* src/bank.py
* src/models/CGAN.py
* src/models/variationalautoencoder.py

The config.py file contains default parameters for the project, including sample information (sampling rate, sample duration), dataset separation sizes and buffer sizes and some autoencoder model parameters.

The bank.py file defines the Bank object which contains all attributes and methods required for loading and preprocessing a dataset. On a standard run it will read all .wav files contained in the given folder which should be structured as:

- bank_example
    * inst1
        * file1.wav
        * file2.wav
        * ...
    * inst2
        * ...
    * ...

When running the bank.preprocess_wav_files() method

## VAE Architecture

The DKVAE project is implemented using a Variational Autoencoder (VAE), a generative model that allows us to create new data samples from the learned distribution of our input data.

The VAE is implemented in TensorFlow, and it's structured in two main parts: the Encoder and the Decoder.

* Encoder: This is the "recognition" part of the model, which takes a drum sound sample as input and encodes it into a compact, continuous latent representation. The encoder is implemented as a Convolutional Neural Network (CNN) that transforms the input drum sound into two parameters in a latent space: a mean and a standard deviation.

* Decoder: This is the "generative" part of the model. The decoder takes a point in the latent space (sampled from the distribution defined by the mean and standard deviation produced by the encoder) and decodes it back into a drum sound. The decoder is implemented as a transposed convolutional neural network, or "deconvolutional" network.

The training objective of a VAE is twofold: It wants the generated samples to look like the training data, and it also wants the points in the latent space to follow a standard normal distribution. To achieve this, we use a combination of two loss functions:

* Reconstruction Loss: Measures how well the decoder is able to reconstruct the input drum sound from the latent representation. We use mean squared error (MSE) as our reconstruction loss.
* KL Divergence Loss: Measures how much the distribution of latent points differs from a standard normal distribution. This encourages the model to use the entire latent space, which helps to avoid overfitting and makes the model more robust.

To generate a new drum sound, we simply sample a point from the standard normal distribution and pass it through the decoder. The output is a waveform representing a drum sound.

## CGAN Architecture

The DKCGAN model was also tried on early stages on development and its model can be found in the /src/models/CGAN.py file. It implements a Conditional Generative Adversarial Network.

The CGAN consists of two main components: the Generator and the Discriminator, both of which are typically implemented as neural networks. In our DKCGAN, we provide additional categorical input, such as drum type (kick, snare, hi-hat, etc.), to both networks, enabling conditioned generation and discrimination.

* Generator: The generator's role is to generate synthetic drum sound samples from a noise vector and a drum type label. The generator is often a deconvolutional neural network when dealing with audio data. The goal of the generator is to generate drum sounds that are so realistic that the discriminator can't tell them apart from real drum sounds.

* Discriminator: The discriminator's role is to distinguish between real drum sound samples and those generated by the generator. It takes both a drum sound and a drum type label as input, and outputs a probability that the given sound sample is real. The discriminator is often implemented as a convolutional neural network for audio data.

During training, the discriminator tries to correctly classify the real and generated drum sounds, thus it is trained to minimize its classification error. The generator tries to fool the discriminator by generating drum sounds that are as realistic as possible, thus it is trained to maximize the discriminator's classification error on generated sounds.

The training alternates between these two objectives, gradually improving both the discriminator's ability to distinguish real sounds from fake ones, and the generator's ability to create realistic sounds.

The training process is guided by the Binary Cross Entropy (BCE) loss, which measures the error between the discriminator's predictions and the actual labels (1 for real sounds and 0 for generated sounds).

After training, we can use the generator to create new drum sounds. To do this, we feed it a random noise vector along with a drum type label. The generator then outputs a synthetic drum sound corresponding to the provided drum type.

In this specific architecture, fooling the discriminator was not achieved and it would always properly classify real and artificial sounds, so this approach was discarded.

# Dataset

# Results

## Performance