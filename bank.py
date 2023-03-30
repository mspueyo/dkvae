import os
import logging

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from timbral_models import timbral_extractor

from config import *


class Bank:
    """
    Reading audio files, extracting features and dumping them.
    Attributes:
        id:             Bank Name
        original_dir:   Path to original folder.
        dir:            Path to encoded folder
        instruments:    List of instruments (Sample Categories)
        timbre:         List of dictionaries of timbral parameters of the samples.
    """
    
    
    def __init__(self, id):
        """Creates Audio object"""
        self.id = id
        self.original_dir, self.dir, self.instruments = Bank.get_dirs(id)
        
        logging.info("DataLoader: Loading dataset {}".format(self.id))
        if not self.is_processed():
            self.timbre = []
            self.data = []
            self.preprocess_wav_files()
            self.dump_timbre()
            self.dump_data()
    
    
    @staticmethod
    def get_dirs(id):
        """Retrieves file and destination folders based on id. Raises exception if bank not found."""
        # Source bank
        src = os.path.join(DATA_PATH, id)
        if not os.path.exists(src):
            raise Exception("Source bank {} does not exist.".format(src))
        
        instruments = [x for x in os.listdir(src) if os.path.isdir(os.path.join(src, x))]
        
        # Make destination bank for processed files
        dst = os.path.join(DATA_PATH, "{}{}".format(id, OUT_BANK_PREFIX))
        if not os.path.exists(dst):
            os.makedirs(dst)
            for inst in instruments:
                os.makedirs(os.path.join(dst, inst))
        
        return src, dst, instruments
    
    
    def preprocess_wav_files(self):
        """Main Loop to execute for each wave file."""
        for inst in self.instruments:
            # Local directories for each sample
            src_dir = os.path.join(self.original_dir, inst)
            dst_dir = os.path.join(self.dir, inst)
            for file in os.listdir(src_dir):
                if file.endswith('.wav'):
                    # Local sample name
                    src_file = os.path.join(src_dir, file)
                    dst_file = os.path.join(dst_dir, file)
                    # Load file
                    sample, _ = librosa.load(src_file, sr=SAMPLE_RATE, mono=True, dtype=np.float64)
                    # Pre process
                    sample = np.trim_zeros(sample, trim='f')  # Remove leading zeros
                    sample = Bank.match_length(sample) # Make all samples equal length
                    # Save file
                    sf.write(dst_file, sample, SAMPLE_RATE)
                    # Get parameters
                    self.timbre.append(Bank.get_timbre(dst_file))
                    self.data.append({
                        "sample": file,
                        "path": dst_file,
                        "original_path": src_file,
                        "label": inst
                    })
    
    
    @staticmethod
    def match_length(audio):
        """Makes audio size match SAMPLE_LEN"""
        if len(audio) > SAMPLE_LEN:
            return audio[0:SAMPLE_LEN]
        elif len(audio) < SAMPLE_LEN:
            return np.pad(audio, (0, SAMPLE_LEN-len(audio)), constant_values=.0)
        return audio
    
    
    @staticmethod
    def get_timbre(path):
        return timbral_extractor(path, verbose=False)


    def dump_timbre(self):
        timbre_df = pd.DataFrame(self.timbre)
        dst_file = os.path.join(self.dir, TIMBRAL_DATA_FILE)
        timbre_df.to_csv(dst_file, index=False)
    
    
    def dump_data(self):
        data_df = pd.DataFrame(self.data)
        dst_file = os.path.join(self.dir, SAMPLE_DATA_FILE)
        data_df.to_csv(dst_file, index=False)
        
        
    def is_processed(self):
        data_df_path = os.path.join(self.dir, SAMPLE_DATA_FILE)
        timbre_df_path = os.path.join(self.dir, TIMBRAL_DATA_FILE)
        if os.path.exists(data_df_path) and os.path.exists(timbre_df_path):
            data_df = pd.read_csv(data_df_path)
            timbre_df = pd.read_csv(timbre_df_path)
            if len(data_df.index) == len(timbre_df.index):
                for i, r in data_df.iterrows():
                    if not os.path.exists(r['path']):
                        return False
                return True
        return False
        
          
if __name__ == "__main__":
    test = Bank("test_bank_big")