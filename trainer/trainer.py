import os
import torch
from torch import nn, optim

from utils.training_losses import mse_with_positive_pressure


# Training class
class Trainer:
    def __init__(self, model, dataloader, learning_rate=0.01):
        self.model = model.double()  # Ensure model parameters are in float64
        self.dataloader = dataloader
        self.criterion_pitch = nn.CrossEntropyLoss()
        self.criterion_step = mse_with_positive_pressure
        self.criterion_duration = mse_with_positive_pressure
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, epochs, callbacks=[]):
        self.model.train()
        history = {'loss': []}

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.dataloader:
                inputs, targets = self._move_to_device(inputs, targets)

                # Forward pass
                pitch_pred, step_pred, duration_pred = self.model(inputs)

                # Compute losses
                loss_pitch = self.criterion_pitch(pitch_pred.double(), targets['pitch'].long())
                loss_step = self.criterion_step(targets['step'].double(), step_pred.double())
                loss_duration = self.criterion_duration(targets['duration'].double(), duration_pred.double())

                # Apply loss weights
                loss = 0.05 * loss_pitch + loss_step + loss_duration

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            history['loss'].append(avg_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            # Callbacks (e.g., checkpointing, early stopping)
            for callback in callbacks:
                callback(epoch, self.model, avg_loss)

        return history

    def evaluate(self):
        self.model.eval()
        losses = {'loss': 0, 'pitch': 0, 'step': 0, 'duration': 0}
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = self._move_to_device(inputs, targets)

                pitch_pred, step_pred, duration_pred = self.model(inputs)

                loss_pitch = self.criterion_pitch(pitch_pred.double(), targets['pitch'].long())
                loss_step = self.criterion_step(targets['step'].double(), step_pred.double())
                loss_duration = self.criterion_duration(targets['duration'].double(), duration_pred.double())

                loss = 0.05 * loss_pitch + loss_step + loss_duration

                losses['pitch'] += loss_pitch.item()
                losses['step'] += loss_step.item()
                losses['duration'] += loss_duration.item()
                losses['loss'] += loss.item()

        for key in losses:
            losses[key] /= len(self.dataloader)

        return losses

    def _move_to_device(self, inputs, targets):
        for key in targets:
            targets[key] = targets[key].double().to(self.device)  # Ensure targets are in float64
        inputs = inputs.double().to(self.device)  # Ensure inputs are in float64
        return inputs, targets


# Callback for early stopping and saving checkpoints
class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=5, filepath='./training_checkpoints/model_checkpoint.pth'):
        self.patience = patience
        self.filepath = filepath
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

        # Ensure the directory exists
        checkpoint_dir = os.path.dirname(self.filepath)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def __call__(self, epoch, model, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), self.filepath)
            print(f"Model checkpoint saved at epoch {epoch + 1}")
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                model.load_state_dict(torch.load(self.filepath, weights_only=True))
                return True  # Indicate early stopping
        return False


def load_checkpoint(model: torch.nn.Module,
                    checkpoint_path: str = './training_checkpoints/model_checkpoint.pth'):
    """Load a specific checkpoint from the specified path into the model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the checkpoint into the model
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda')))
    print(f"Loaded checkpoint from '{checkpoint_path}' successfully.")

    return model
