![](https://i.imgur.com/pnRo5b2.png)

DK·VAE (Drumkit Variational Autoencoder) is an audio generation tool based on variational autoencoders. It has been developed as a final thesis for the Data Science MSc degree at UOC (Universitat Oberta de Catalunya).

The DKVAE project uses a Variational Autoencoder (VAE), a type of generative model, to learn a compact, continuous representation of drum sounds. The model is trained on a large dataset of drum sounds, and it can generate novel drum sounds that are similar to the training data. DrumKitVAE can be seen as a drum machine that has the ability to generate an infinite number of unique drum sounds, adding creativity and versatility to your music production process.


# Table of Contents


# Installation

## Requirements
DKVAE depends on the following libraries:


## Notebook
You can try DKVAE online with no installation on the following Google Colab notebook.

## Installation

Alternatively, the package can be installed locally by running the following comand:

```
pip install https://github.com/mspueyo/dkvae.git
```

# Implementation

The project consists on a flexible and scalable architecture that includes audio preprocessing techniques and sound bank information in objects that are able to interact within them. The most important objects that are included are:

* config.py
* src/bank.py
* src/models/variationalautoencoder.py
* src/models/autoencoder.py
* src/models/CGAN.py

The config.py file contains default parameters for the project, including sample information (sampling rate, sample duration), dataset separation sizes and buffer sizes and some autoencoder model parameters.

## Bank

The Bank class is a module for processing audio data. It reads audio files, extracts features and saves them for further use. This class is especially useful for preprocessing and handling datasets for machine learning tasks related to audio data.

At initialization it will read the passed data folder which should be structured as:

> - bank_example
>    * instrument1
>        * file1.wav
>        * file2.wav
>        * ...
>    * instrument2
>        * ...
>    * ...

The class main attributes are:

* id: The name of the bank, derived from the directory name.
* original_dir: The path to the original directory, where raw audio files are stored.
* dir: The path to the encoded directory, where preprocessed data is saved.
* instruments: A list of instruments or sample categories available in the bank.
* data: A pandas DataFrame containing information about each audio sample.

The Bank class includes several key methods for handling audio data:

* Initialization: The __init__ method creates an instance of the Bank class with the specified directory of raw audio files. If the audio files haven't been processed yet, the method initiates the preprocessing step.

* Directories Handling: The get_dirs method retrieves the source and destination directories based on the bank ID and prepares them for further processing. It also logs the number of found instruments (sample categories) in the bank.

* Audio Preprocessing: The preprocess_wav_files method executes a set of operations on each audio file in the source directory: it loads the audio file, trims leading zeros, adjusts the length of the audio to a predefined sample length, computes a log spectogram of the audio, normalizes the spectogram, and saves the normalized spectogram as a numpy file.

* The is_processed method checks if the bank has already been processed and if all files referenced in the CSV data file exist.

* The read_data method loads the CSV data file.

* Spectrogram Loading: The load_spectograms method loads the spectrograms of all the audio samples and returns them as numpy arrays divided into training, testing, and validation datasets.

* Utility Functions: The class also includes several static utility methods for common operations on audio data, including train-test-validation split (test_train_val), matching the audio length to the configured sample length (match_length), computing a log spectrogram of an audio (get_log_spectogram), converting a spectrogram back into an audio signal (spectogram_to_audio), and normalizing a numpy array (normalize).

This class relies on various libraries such as librosa, numpy, pandas, tensorflow and soundfile to handle audio files, extract features, and perform various operations on the data.

## Variational Autoencoder Model

The DKVAE model architecture is based in a Variational Autoencoder (VAE), a generative model that allows us to create new data samples from the learned distribution of our input data. The VAE is implemented in TensorFlow, and it's structured in two main parts: the Encoder and the Decoder.

* Encoder: This is the "recognition" part of the model, which takes a drum sound sample as input and encodes it into a compact, continuous latent representation. The encoder is implemented as a Convolutional Neural Network (CNN) that transforms the input drum sound into two parameters in a latent space: a mean and a standard deviation.

* Decoder: This is the "generative" part of the model. The decoder takes a point in the latent space (sampled from the distribution defined by the mean and standard deviation produced by the encoder) and decodes it back into a drum sound. The decoder is implemented as a transposed convolutional neural network, or "deconvolutional" network.

The training objective of a VAE is for the generated samples to look like the training data, and it also wants the points in the latent space to follow a standard normal distribution. To achieve this, we use a combination of two loss functions:

* Reconstruction Loss: Measures how well the decoder is able to reconstruct the input drum sound from the latent representation. We use mean squared error (MSE) as our reconstruction loss.
* KL Divergence Loss: Measures how much the distribution of latent points differs from a standard normal distribution. This encourages the model to use the entire latent space, which helps to avoid overfitting and makes the model more robust.

To generate a new drum sound, we simply sample a point from the standard normal distribution and pass it through the decoder. The output is a waveform representing a drum sound.

## Autoencoder Model

The Autoencoder model is a type of artificial neural network used for learning efficient codings of input data. Similar to the Variational Autoencoder (VAE), it is also implemented in TensorFlow and consists of two main components: an Encoder and a Decoder.

* Encoder: The Encoder, also known as the "recognition" part of the model, accepts input data and compresses it into a compact latent representation. However, unlike the VAE which maps the input to a distribution defined by mean and standard deviation parameters, the Encoder in an Autoencoder maps the input data to a fixed point in the latent space. This Encoder is designed using a Convolutional Neural Network (CNN) for tasks involving audio spectograms, and Dense layers for simple tabular data.

* Decoder: The Decoder, or the "generative" part of the model, takes the latent representation produced by the Encoder and reconstructs the original input data. It uses a network architecture that mirrors the Encoder. In this case, it is implemented as a transposed convolutional neural network, or "deconvolutional" network.

The training objective of an Autoencoder is to minimize the difference between the input and the reconstructed output. This objective is captured by the Reconstruction Loss, which measures how well the Decoder is able to reconstruct the input from the latent representation. Mean Squared Error (MSE) is commonly used as the Reconstruction Loss.

Unlike VAEs, standard Autoencoders don't have a component equivalent to the KL Divergence Loss. This is because they don't aim to make the latent space follow any particular distribution. However, this may sometimes result in a less efficient use of the latent space, where much of it may be left unutilized.

To generate a representation of the original data (such as a compressed version of an image, or a denoised version of a sound), the input data is passed through the Encoder to the Decoder. The output of the Decoder is a reconstruction of the original data based on the learned features in the latent space.

This model is deprecated and not maintained as it evolved to be a Variational Autoencoder which is more robust.

## CGAN Model

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