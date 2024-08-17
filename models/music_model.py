from torch import nn
from torch.utils.data import DataLoader

from dataloaders.dataloaders import MusicDataLoader
from datasets.datasets import SequenceDataset, MusicDataset
from trainer.trainer import Trainer


class MusicModel(nn.Module):
    def __init__(self, input_shape, lstm_units=128,  dropout_rate=0.3):
        super(MusicModel, self).__init__()
        self.lstm = nn.LSTM(input_shape[1], lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.pitch = nn.Linear(lstm_units, 128)
        self.step = nn.Linear(lstm_units, 1)
        self.duration = nn.Linear(lstm_units, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        pitch = self.pitch(x)
        step = self.step(x)
        duration = self.duration(x)
        return pitch, step, duration

if __name__ == '__main__':
    data_dir = '../data/maestro-v2.0.0'
    music_dataset = MusicDataset(data_dir=data_dir, num_files=5)

    # Convert the notes to a PyTorch tensor
    notes_tensor = music_dataset.get_notes_tensor()

    # Use the notes_tensor to create a SequenceDataset, similar to TensorFlow's create_sequences
    seq_length = 25
    vocab_size = 128
    sequence_dataset = SequenceDataset(notes_tensor, seq_length=seq_length, vocab_size=vocab_size)

    music_data_loader = MusicDataLoader(sequence_dataset, batch_size=64, num_workers=8)
    train_loader = music_data_loader.get_data_loader()
    input_shape = (seq_length, 3)
    model = MusicModel(input_shape)
    trainer = Trainer(model, train_loader, learning_rate=0.005)
