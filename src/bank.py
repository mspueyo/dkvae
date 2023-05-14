import os
import logging
import random

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import audio_dataset_from_directory

from config import *


class Bank:
    """
    Reading audio files, extracting features and dumping them.
    Attributes:
        id:             Bank Name.
        original_dir:   Path to original folder.
        dir:            Path to encoded folder.
        instruments:    List of instruments (Sample Categories).
        data:           DataFrame containing sample information.
    """
    
    def __init__(self, original_dir, to_spec=True):
        """Creates Audio object
        :param original_dir: path to bank"""
        self.id = original_dir.split('/')[-1]
        self.get_dirs(original_dir)
        if not self.is_processed():
            logging.info(f"{self.id} Not processed. Processing bank...")
            self.df = []
            self.preprocess_wav_files()
            self.dump_data()
        else:
            logging.info(f"{self.id} Already processed.")
    
    def get_dirs(self, original_dir):
        """Retrieves file and destination folders based on id. Raises exception if bank not found.
        Returns:
        :src: original directory.
        :dst: destination directory."""
        data_path = os.path.dirname(original_dir)
        # Source bank
        src = os.path.join(data_path, self.id)
        logging.info(f"Seeking source bank {src}")
        if not os.path.exists(src):
            logging.error("The path specified could not be found. If you are running in colab verify that your disk is mounted.")
            raise Exception(f"Source bank {src} does not exist.")
        
        instruments = [x for x in os.listdir(src) if os.path.isdir(os.path.join(src, x))]
        logging.info(f"Found {len(instruments)} instruments: {instruments}")
        
        # Make destination bank for processed files
        dst = os.path.join(data_path, "{}{}".format(self.id, OUT_BANK_SUFFIX))
        logging.info(f"Creating destination bank {dst}...")
        train_seq = ['test', 'train', 'val']
        if not os.path.exists(dst):
            os.makedirs(dst)
            for t in train_seq:
                    for inst in instruments:
                        os.makedirs(os.path.join(dst, t, inst))
        self.original_dir = src
        self.dir = dst
        self.instruments = instruments
    
    def preprocess_wav_files(self):
        """Main Loop to execute for each wav file."""
        for inst in self.instruments:
            # Local directories for each sample
            src_dir = os.path.join(self.original_dir, inst)
            for file in os.listdir(src_dir):
                if file.endswith('.wav'):
                    # Local sample name
                    src_file = os.path.join(src_dir, file)

                    # Train, Test, Validation Split
                    split = self.test_train_val()
                    dst_dir = os.path.join(self.dir, split, inst)
                    dst_file = os.path.join(dst_dir, f"{file.split('.')[0]}.npy")

                    # Load file
                    logging.debug(f"Preprocessing {src_file}...")
                    try:
                        sample, _ = librosa.load(src_file, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
                    except Exception:
                        logging.warning(f"{file} could not be loaded. Skipping file.")
                    else:
                        # Pre process
                        sample = np.trim_zeros(sample, trim='f')  # Remove leading zeros
                        sample = Bank.match_length(sample) # Make all samples equal length
                        log_spectogram = Bank.get_log_spectogram(sample)
                        normalized_spectogram = Bank.normalize(log_spectogram, 0, 1)
                        # Save file
                        if log_spectogram.min() != log_spectogram.max():
                            logging.info("Saving file...")
                            np.save(dst_file, normalized_spectogram)
                            # Get parameters
                            self.df.append({
                                "sample": file,
                                "path": dst_file,
                                "original_path": src_file,
                                "label": inst,
                                "split": split,
                                "min_value": log_spectogram.min(),
                                "max_value": log_spectogram.max()
                            })
        self.df = pd.DataFrame(self.df)

    def dump_data(self):
        """Saves sample data to csv."""
        dst_file = os.path.join(self.dir, SAMPLE_DATA_FILE)
        logging.info(f"Saving data to {dst_file}")
        self.df.to_csv(dst_file, index=False)
           
    def is_processed(self):
        """Checks if sample file has already been processed and all files stored in csv exists."""
        data_df_path = os.path.join(self.dir, SAMPLE_DATA_FILE)
        if os.path.exists(data_df_path):
            self.df = pd.read_csv(data_df_path)
            for i, r in self.df.iterrows():
                if not os.path.exists(r['path']):
                    return False
                return True
        return False

    def read_data(self):
        """Loads sample data file."""
        file = os.path.join(self.dir, SAMPLE_DATA_FILE)
        logging.debug(f"Reading data from {file}...")
        self.df = pd.read_csv(file)
        self.df = self.df[self.df['min_value'] != self.df['max_value']]

    def load_spectograms(self):
        """Loads spectograms."""
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_val = []
        y_val = []

        logging.info("Loading audio spectograms...")
        for i, r in self.df.iterrows():
            file_path = r.path
            label = r.label
            split = r.split
            spec = np.load(file_path)
            if r.min_value != r.max_value and split == "train":
                x_train.append(spec)
                y_train.append(label)
            elif r.min_value != r.max_value and split == "test":
                x_test.append(spec)
                y_test.append(label)
            elif r.min_value != r.max_value and split == "val":
                x_val.append(spec)
                y_val.append(label)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        x_train = x_train[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        y_test = y_test[..., np.newaxis]
        x_val = x_val[..., np.newaxis]
        y_val = y_val[..., np.newaxis]

        return x_train, y_train, x_test, y_test, x_val, y_val



    @staticmethod
    def test_train_val():
        rnd = random.random()
        if rnd <= VAL_SIZE:
            return 'val'
        elif rnd <= VAL_SIZE + TEST_SIZE:
            return 'test'
        return 'train'

    @staticmethod
    def match_length(audio):
        """Matches audio length to config file."""
        if len(audio) > SAMPLE_LEN:
            return audio[0:SAMPLE_LEN]
        elif len(audio) < SAMPLE_LEN:
            return np.pad(audio, (0, SAMPLE_LEN-len(audio)), constant_values=.0)
        return audio

    @staticmethod
    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
    @staticmethod
    def get_log_spectogram(audio):
        logging.info("Computing spectogram...")
        stft = librosa.stft(audio, n_fft=FFT_FRAME_SIZE, hop_length=FFT_HOP_LENGTH)[:-1, :]
        log_spectogram = librosa.amplitude_to_db(np.abs(stft))
        return log_spectogram
    
    @staticmethod
    def spectogram_to_audio(log_spectogram):
        stft = librosa.db_to_amplitude(log_spectogram)
        audio = librosa.griffinlim(stft, hop_length=FFT_HOP_LENGTH, n_iter=64)
        return audio
    
    @staticmethod
    def normalize(arr, min, max):
        logging.info("Normalizing spectogram...")
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min())
        norm_arr = norm_arr * (max - min) + min
        return norm_arr