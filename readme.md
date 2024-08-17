---

# Music Generation with LSTM in PyTorch

This project implements a music generation model using Long Short-Term Memory (LSTM) networks in PyTorch. The model is trained on MIDI data from the MAESTRO dataset to predict sequences of musical notes, including pitch, step, and duration.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Generating Music](#generating-music)
- [Checkpoints](#checkpoints)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to generate sequences of music using an LSTM-based model. The model is trained on sequences of musical notes and learns to predict the next note in a sequence, enabling it to generate coherent musical pieces.

## Features

- **LSTM-based Model**: Utilizes LSTM networks to model temporal dependencies in music sequences.
- **Custom Loss Function**: Includes a custom loss function with positive pressure to ensure stable training.
- **Training with Checkpoints**: Supports early stopping and model checkpointing during training.
- **MIDI File Generation**: Generates MIDI files from predicted note sequences.

## Installation

### Requirements

- Python 3.7 or higher
- PyTorch 1.8.0 or higher
- Additional Python packages:
  - `numpy`
  - `pandas`
  - `pretty_midi`
  - `matplotlib`
  - `torch`
  - `sounddevice`

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/music-generation-lstm.git
    cd music-generation-lstm
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

### Download the MAESTRO Dataset

The model is trained on the MAESTRO v2.0.0 dataset, which contains MIDI files for classical piano performances.

1. Download the dataset from [here](https://magenta.tensorflow.org/datasets/maestro).
2. Extract the dataset into the `data/maestro-v2.0.0/` directory.

### Data Processing

The MIDI files are processed into sequences of musical notes, where each note is represented by pitch, step, and duration. These sequences are then converted into PyTorch tensors for training.

```python
from music_generation import MusicDataset

data_dir = './data/maestro-v2.0.0'
music_dataset = MusicDataset(data_dir=data_dir, num_files=1000)
notes_tensor = music_dataset.get_notes_tensor()
```

## Model Architecture

The model is an LSTM-based network that takes sequences of notes as input and predicts the next note in the sequence.

- **LSTM Layer**: Extracts temporal dependencies from the input sequences.
- **Dense Layers**: Predicts pitch, step, and duration for the next note in the sequence.

### Model Implementation

```python
import torch.nn as nn

class MusicModel(nn.Module):
    def __init__(self, input_shape, lstm_units=128):
        super(MusicModel, self).__init__()
        self.lstm = nn.LSTM(input_shape[1], lstm_units, batch_first=True)
        self.pitch = nn.Linear(lstm_units, 128)
        self.step = nn.Linear(lstm_units, 1)
        self.duration = nn.Linear(lstm_units, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Use the last output of the LSTM
        pitch = self.pitch(x)
        step = self.step(x)
        duration = self.duration(x)
        return pitch, step, duration
```

## Training

### Training Process

The model is trained using a custom `Trainer` class that handles the training loop, loss calculation, and optimization.

```python
from music_generation import Trainer, EarlyStoppingAndCheckpoint

trainer = Trainer(model, train_loader, learning_rate=0.005)
callbacks = [EarlyStoppingAndCheckpoint(filepath='./training_checkpoints/model_checkpoint.pth')]
history = trainer.train(epochs=50, callbacks=callbacks)
```

### Loss Functions

The model uses the following loss functions:
- **CrossEntropyLoss** for pitch prediction.
- **Mean Squared Error** (MSE) with positive pressure for step and duration prediction.

## Evaluation

After training, the model can be evaluated on a validation dataset to check its performance.

```python
evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")
```

## Generating Music

You can generate music sequences using the trained model and save them as MIDI files.

```python
from music_generation import generate_notes

generated_notes = generate_notes(model, raw_notes, seq_length=25, vocab_size=128, instrument_name='Acoustic Grand Piano')
```

## Checkpoints

The model supports checkpointing to save and load training progress.

### Saving Checkpoints

Checkpoints are automatically saved during training if the model improves.

### Loading Checkpoints

To load the latest checkpoint and continue training or evaluation:

```python
from music_generation import load_checkpoint

model = load_checkpoint(model, checkpoint_path='./training_checkpoints/model_checkpoint.pth')
```

## Usage

### Running the Training

To start training, run:

```bash
python train.py
```

### Generating MIDI Files

To generate MIDI files from a trained model:

```bash
python generate_music.py
```

## Contributing

We welcome contributions to improve this project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear explanation of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

