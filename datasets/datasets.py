import glob
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.notes_conv import midi_to_notes

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
_SAMPLING_RATE = 16000

class SequenceDataset(Dataset):
    def __init__(self, data_tensor, seq_length, vocab_size=128):
        self.data = data_tensor
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Get the input sequence
        input_sequence = self.data[idx:idx + self.seq_length]

        # Normalize note pitch
        input_sequence[:, 0] /= self.vocab_size

        # The target sequence is the same sequence but shifted by one time step
        target_sequence = self.data[idx + 1:idx + 1 + self.seq_length]

        # Create target dictionary
        targets = {key: target_sequence[:, i] for i, key in enumerate(['pitch', 'step', 'duration'])}

        return input_sequence, targets

class MusicDataset:
    def __init__(self, data_dir: str, num_files: int = 5):
        self.data_dir = pathlib.Path(data_dir)
        self.num_files = num_files
        self.filenames = glob.glob(str(self.data_dir / '**/*.mid*'))
        self.all_notes = self._load_notes()

    def _load_notes(self) -> pd.DataFrame:
        all_notes = []
        for f in self.filenames[:self.num_files]:
            notes = midi_to_notes(f)
            all_notes.append(notes)
        return pd.concat(all_notes, ignore_index=True)

    def get_notes_tensor(self, key_order=None):
        if key_order is None:
            key_order = ['pitch', 'step', 'duration']
        return torch.tensor(np.stack([self.all_notes[key] for key in key_order], axis=1), dtype=torch.float32)

if __name__ == '__main__':
    data_dir = '../data/maestro-v2.0.0'
    music_dataset = MusicDataset(data_dir=data_dir, num_files=5)

    # Convert the notes to a PyTorch tensor
    notes_tensor = music_dataset.get_notes_tensor()
    print("Shape of the first note tensor:", notes_tensor[0].shape)

    # Use the notes_tensor to create a SequenceDataset
    seq_length = 25
    vocab_size = 128
    sequence_dataset = SequenceDataset(notes_tensor, seq_length=seq_length, vocab_size=vocab_size)

    # Create a DataLoader for the sequence dataset
    data_loader = DataLoader(sequence_dataset, batch_size=64, shuffle=False)

    # Take the first batch (equivalent to `take(1)` in TensorFlow)
    for seq, target in data_loader:
        print('Input sequence shape:', seq.shape)
        print('Target pitch shape:', target['pitch'].shape)
        print('Target step shape:', target['step'].shape)
        print('Target duration shape:', target['duration'].shape)
        break