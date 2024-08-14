import collections
import os
import glob
import pathlib
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pretty_midi
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import random

from dataloaders.dataloaders import MusicDataLoader
from datasets.datasets import SequenceDataset, MusicDataset
from models.music_model import MusicModel
from trainer.trainer import Trainer, EarlyStoppingAndCheckpoint, load_checkpoint
from utils.download_dataset import download_and_extract_maestro
from utils.notes_conv import notes_to_midi, midi_to_notes
from utils.training_losses import mse_with_positive_pressure

# Constants
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
_SAMPLING_RATE = 16000


def generate_notes(
        model: torch.nn.Module,
        raw_notes: pd.DataFrame,
        seq_length: int,
        vocab_size: int,
        instrument_name: str,
        temperature: float = 2.0,
        num_predictions: int = 120,
        out_file: str = 'output.mid'
):
    """
    Generates a sequence of notes using the trained model and saves them as a MIDI file.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        raw_notes (pd.DataFrame): DataFrame containing the raw notes.
        seq_length (int): The sequence length used for input to the model.
        vocab_size (int): The vocabulary size (number of unique pitches).
        key_order (list): List of keys representing ['pitch', 'step', 'duration'].
        instrument_name (str): The name of the instrument for the MIDI file.
        temperature (float): Temperature for controlling the randomness of predictions.
        num_predictions (int): Number of notes to generate.
        out_file (str): Path to save the generated MIDI file.

    Returns:
        pd.DataFrame: DataFrame containing the generated notes.
    """

    key_order = ['pitch', 'step', 'duration']

    # Sample notes as the initial sequence
    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # Normalize the initial sequence of notes
    input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])

    # List to hold generated notes
    generated_notes = []
    prev_start = 0

    # Generate new notes
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))

        # Update the input sequence for the next prediction
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    # Convert generated notes to DataFrame
    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

    # Save the generated notes as a MIDI file
    notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)

    return generated_notes


def predict_next_note(
        notes: np.ndarray,
        model: torch.nn.Module,
        temperature: float = 1.0
) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration) using a trained sequence model."""

    assert temperature > 0

    # Convert to tensor and add batch dimension
    inputs = torch.tensor(notes, dtype=torch.float32).unsqueeze(0)
    inputs = inputs.to(next(model.parameters()).device)  # Move to model's device

    # Forward pass
    model.eval()
    with torch.no_grad():
        pitch_logits, step, duration = model(inputs)

    # Adjust temperature
    pitch_logits /= temperature

    # Sample pitch from the logits
    pitch_probs = F.softmax(pitch_logits, dim=-1)
    pitch = torch.multinomial(pitch_probs, num_samples=1)
    pitch = pitch.squeeze().item()

    # Squeeze the outputs
    step = step.squeeze().item()
    duration = duration.squeeze().item()

    # Ensure `step` and `duration` are non-negative
    step = max(0, step)
    duration = max(0, duration)

    return int(pitch), float(step), float(duration)


# Example usage
if __name__ == '__main__':
    # Download and extract the MAESTRO dataset
    data_dir = 'data/maestro-v2.0.0'
    maestro_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip'
    download_and_extract_maestro(data_dir, maestro_url)

    # Initialize the dataset
    music_dataset = MusicDataset(data_dir=data_dir, num_files=100)
    notes_tensor = music_dataset.get_notes_tensor()

    # Create the sequence dataset
    seq_length = 25
    seq_ds = SequenceDataset(notes_tensor, seq_length=seq_length)

    # Initialize the data loader with multiprocessing support
    music_data_loader = MusicDataLoader(seq_ds, batch_size=64, num_workers=8)
    train_loader = music_data_loader.get_data_loader()
    input_shape = (seq_length, 3)  # Adjust according to your data

    model = MusicModel(input_shape)

    # Initialize the trainer
    trainer = Trainer(model, train_loader, learning_rate=0.005)

    # Define callbacks
    callbacks = [EarlyStoppingAndCheckpoint(filepath='./training_checkpoints/model_checkpoint.pth')]

    # Train the model
    history = trainer.train(epochs=50, callbacks=callbacks)

    # Plot training loss
    plt.plot(range(len(history['loss'])), history['loss'], label='total loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    # # Evaluate the model
    losses = trainer.evaluate()
    print(f"Evaluation losses: {losses}")
    #
    filenames = glob.glob(str(pathlib.Path(data_dir) / '**/*.mid*'))
    print('Number of files:', len(filenames))
    sample_file = filenames[1]

    pm = pretty_midi.PrettyMIDI(sample_file)
    raw_notes = midi_to_notes(sample_file)
    seq_length = 100
    vocab_size = 128
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    input_shape = (seq_length, 3)  # Adjust according to your data
    model = MusicModel(input_shape)
    model = load_checkpoint(model)


    generated_notes = generate_notes(
        model=model,
        raw_notes=raw_notes,
        seq_length=seq_length,
        vocab_size=vocab_size,
        instrument_name=instrument_name,
        temperature=50.0,
        num_predictions=100,
        out_file='output.mid'
    )
