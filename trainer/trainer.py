import os
import torch
from torch import nn, optim

from utils.training_losses import mse_with_positive_pressure


# Training class
import os
import torch
from torch import nn, optim

from utils.training_losses import mse_with_positive_pressure

# Training class
class Trainer:
    def __init__(self, model, dataloader, learning_rate=0.01, checkpoint_dir='./training_checkpoints'):
        self.model = model.double()  # Ensure model parameters are in float64
        self.dataloader = dataloader
        self.criterion_pitch = nn.CrossEntropyLoss()
        self.criterion_step = mse_with_positive_pressure
        self.criterion_duration = mse_with_positive_pressure
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir

        # Load the latest checkpoint if it exists
        self.load_latest_checkpoint()

    def load_latest_checkpoint(self):
        """Load the latest checkpoint into the model if it exists."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint from '{checkpoint_path}' successfully.")
        else:
            print("No checkpoint found. Starting training from scratch.")

    def train(self, epochs, callbacks=[]):
        self.model.train()  # Set the model to training mode
        history = {'loss': []}

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.dataloader:
                inputs, targets = self._move_to_device(inputs, targets)

                # Forward pass
                pitch_pred, step_pred, duration_pred = self.model(inputs)

                # Reshape pitch predictions and targets for loss computation
                pitch_pred = pitch_pred.view(-1, 128)  # Flatten the pitch predictions
                targets_pitch = targets['pitch'].view(-1)  # Flatten the pitch targets

                # Reshape step and duration predictions and targets
                step_pred = step_pred.view(-1)
                duration_pred = duration_pred.view(-1)
                targets_step = targets['step'].view(-1)
                targets_duration = targets['duration'].view(-1)

                # Compute losses
                loss_pitch = self.criterion_pitch(pitch_pred.float(), targets_pitch.long())
                loss_step = self.criterion_step(step_pred.float(), targets_step.float())
                loss_duration = self.criterion_duration(duration_pred.float(), targets_duration.float())

                # Apply loss weights
                loss = loss_pitch + loss_step + loss_duration

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Calculate the average loss for this epoch
            avg_loss = total_loss / len(self.dataloader)
            history['loss'].append(avg_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            # Callbacks (e.g., checkpointing, early stopping)
            for callback in callbacks:
                callback(epoch, self.model, avg_loss)

        return history

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        losses = {'loss': 0, 'pitch': 0, 'step': 0, 'duration': 0}
        num_batches = len(self.dataloader)  # Get the number of batches

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for inputs, targets in self.dataloader:
                inputs, targets = self._move_to_device(inputs, targets)

                # Forward pass through the model
                pitch_pred, step_pred, duration_pred = self.model(inputs)

                # Reshape predictions and targets for loss computation
                batch_size, seq_length, _ = pitch_pred.shape

                # Reshape pitch predictions and targets to be compatible with CrossEntropyLoss
                pitch_pred = pitch_pred.view(batch_size * seq_length, -1)  # Flatten the pitch predictions
                pitch_target = targets['pitch'].view(-1)  # Flatten the pitch targets

                # Compute pitch loss
                loss_pitch = self.criterion_pitch(pitch_pred, pitch_target.long())

                # Reshape step and duration predictions and targets
                step_pred = step_pred.view(-1)
                duration_pred = duration_pred.view(-1)
                step_target = targets['step'].view(-1)
                duration_target = targets['duration'].view(-1)

                # Compute step and duration losses
                loss_step = self.criterion_step(step_pred.float(), step_target.float())
                loss_duration = self.criterion_duration(duration_pred.float(), duration_target.float())

                # Combine the losses
                total_loss = 0.05 * loss_pitch + loss_step + loss_duration

                # Accumulate losses for averaging
                losses['pitch'] += loss_pitch.item()
                losses['step'] += loss_step.item()
                losses['duration'] += loss_duration.item()
                losses['loss'] += total_loss.item()

        # Compute the average loss over all batches
        for key in losses:
            losses[key] /= num_batches

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
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint from '{checkpoint_path}' successfully.")

    return model
