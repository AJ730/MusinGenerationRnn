import glob
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils.notes_conv import midi_to_notes

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
        self.seq_length = seq_length + 1
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        # Get the sequence
        sequence = self.data[idx:idx + self.seq_length]

        # Normalize note pitch
        sequence[:, 0] /= self.vocab_size

        # Split into input and label
        inputs = sequence[:-1]
        labels_dense = sequence[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(['pitch', 'step', 'duration'])}

        return inputs, labels

# Main classes
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
        return pd.concat(all_notes)

    def get_notes_tensor(self, key_order=None):
        if key_order is None:
            key_order = ['pitch', 'step', 'duration']
        return torch.tensor(np.stack([self.all_notes[key] for key in key_order], axis=1))


if __name__ == '__main__':
    data_dir = '../data/maestro-v2.0.0'
    music_dataset = MusicDataset(data_dir=data_dir, num_files=5)

    # Convert the notes to a PyTorch tensor
    notes_tensor = music_dataset.get_notes_tensor()
    print(notes_tensor[0].shape)

    # Use the notes_tensor to create a SequenceDataset, similar to TensorFlow's create_sequences
    seq_length = 25
    vocab_size = 128
    sequence_dataset = SequenceDataset(notes_tensor, seq_length=seq_length, vocab_size=vocab_size)


    # Create a DataLoader for the sequence dataset
    data_loader = DataLoader(sequence_dataset, batch_size=1, shuffle=False)

    # Take the first batch (equivalent to `take(1)` in TensorFlow)
    for seq, target in data_loader:
        seq = seq.squeeze(0)
        print('sequence shape:', seq.shape)
        print('sequence elements (first 10):', seq[0:10])  # Adjusting for batch dimension
        print()
        print('target:', target)
        break  # Only take the first batch, equivalent to TensorFlow's `take(1)`