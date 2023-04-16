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
    
    
    def __init__(self, id, to_spec=True):
        """Creates Audio object
        :param id: bank name (Subfolder dir name)"""
        self.id = id
        self.get_dirs()
        if not self.is_processed():
            logging.info(f"{self.id} Not processed. Processing bank...")
            self.data = []
            self.preprocess_wav_files()
            self.dump_data()
        else:
            logging.info(f"{self.id} Already processed. Loading data...")
            self.read_data()
        logging.info(f"Reading dataset {self.id}...")
        self.read_dataset()

    
    def get_dirs(self):
        """Retrieves file and destination folders based on id. Raises exception if bank not found.
        Returns:
        :src: original directory.
        :dst: destination directory."""
        # Source bank
        src = os.path.join(DATA_PATH, self.id)
        logging.info(f"Seeking source bank {src}")
        if not os.path.exists(src):
            raise Exception(f"Source bank {src} does not exist.")
        
        instruments = [x for x in os.listdir(src) if os.path.isdir(os.path.join(src, x))]
        logging.info(f"Found {len(instruments)} instruments: {instruments}")
        
        # Make destination bank for processed files
        dst = os.path.join(DATA_PATH, "{}{}".format(self.id, OUT_BANK_SUFFIX))
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
                    rnd = random.random()
                    if rnd <= VAL_SIZE:
                        split = 'val'
                    elif rnd <= VAL_SIZE + TEST_SIZE:
                        split = 'test'
                    else:
                        split = 'train'
                    dst_dir = os.path.join(self.dir, split, inst)
                    dst_file = os.path.join(dst_dir, file)
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
                        # Save file
                        sf.write(dst_file, sample, SAMPLE_RATE)
                        # Get parameters
                        self.data.append({
                            "sample": file,
                            "path": dst_file,
                            "original_path": src_file,
                            "label": inst,
                            "split": split
                        })
    
    
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
    
    
    def dump_data(self):
        """Saves sample data to csv."""
        data_df = pd.DataFrame(self.data)
        dst_file = os.path.join(self.dir, SAMPLE_DATA_FILE)
        logging.info(f"Saving data to {dst_file}")
        data_df.to_csv(dst_file, index=False)
        
        
    def is_processed(self):
        """Checks if sample file has already been processed and all files stored in csv exists."""
        data_df_path = os.path.join(self.dir, SAMPLE_DATA_FILE)
        if os.path.exists(data_df_path):
            self.data = pd.read_csv(data_df_path)
            for i, r in self.data.iterrows():
                if not os.path.exists(r['path']):
                    return False
                return True
        return False


    def read_data(self):
        """Loads sample data file."""
        file = os.path.join(self.dir, SAMPLE_DATA_FILE)
        logging.debug(f"Reading data from {file}...")
        self.data = pd.read_csv(file)

    
    def read_dataset(self):
        """Reads tensorflow dataset."""
        logging.info("Loading train dataset...")
        self.train_ds = audio_dataset_from_directory(directory=os.path.join(self.dir, 'train'), seed=42, output_sequence_length=SAMPLE_LEN)
        logging.info("Loading test dataset...")
        self.test_ds = audio_dataset_from_directory(directory=os.path.join(self.dir, 'test'), seed=42, output_sequence_length=SAMPLE_LEN)
        logging.info("Loading val dataset...")
        self.val_ds = audio_dataset_from_directory(directory=os.path.join(self.dir, 'val'), seed=42, output_sequence_length=SAMPLE_LEN)

        self.train_ds = self.train_ds.map(Bank.squeeze, tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.map(Bank.squeeze, tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.map(Bank.squeeze, tf.data.AUTOTUNE)