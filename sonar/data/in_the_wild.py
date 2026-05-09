import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np 
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import roc_curve
import eval_metrics_DF as em
from data_utils_SSL import process_Rawboost_feature
from RawBoost import normWav
from data_utils_SSL import pad


class In_the_wild_dataset(Dataset):
    """In-the-Wild deepfake corpus loader. Supply paths via CLI / config."""
    def __init__(self, csv_file: str, audio_dir: str, sr: int = 16000, duration: int = 4, transform=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.num_samples = sr * duration + 600 ## the model trained on 64600 samples
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0])
        label = 0 if self.data.iloc[idx, 2] == 'spoof' else 1
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sr)
        filename = self.data.iloc[idx, 0]

        # Ensure audio length is 5 seconds
        audio_pad = pad(audio,self.num_samples)
        audio_pad = normWav(audio_pad,0)

        if self.transform:
            audio_pad = self.transform(audio_pad)

            
        return torch.tensor(audio_pad, dtype=torch.float32), label
    


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id



