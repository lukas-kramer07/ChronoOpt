# src/models/prediction_model.py
# This file will contain the PyTorch-based prediction model (e.g., LSTM)
# that predicts the next day's full set of features, with GPU support and
# training/validation split, now including a learning rate scheduler,
# per-metric error calculation, and regularization techniques (Dropout, Weight Decay, Gradient Clipping).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import the scheduler
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error # Import MAE

# 1. Device Configuration: Check for CUDA (GPU) availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PredictionModel(nn.Module):
    """
    A PyTorch-based prediction model (e.g., LSTM) that predicts the next day's
    full set of features based on a sequence of past daily features.
    Supports GPU acceleration if available.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1,
                 dropout_rate: float = 0.25): # Hardcoded default dropout_rate
        """
        Initializes the PredictionModel.

        Args:
            input_size (int): The number of features per day in the input sequence.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            output_size (int): The number of features to predict for the next day.
                               This should be equal to input_size.
            num_layers (int): Number of recurrent layers.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(PredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device # Store the device for easy access within the model

        # Define the LSTM layer
        # Added dropout to LSTM if num_layers > 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

        # Move the entire model to the selected device (CPU or GPU)
        self.to(self.device)

        print(f"PredictionModel initialized on {self.device} with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}, dropout_rate={dropout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the prediction model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Predicted features for the next day, shape (batch_size, output_size).
        """
        # Initialize hidden and cell states on the same device as the input tensor 'x'
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply dropout to the output of the LSTM layer (only the last time step)
        out = self.dropout(out[:, -1, :])

        out = self.fc(out) # Pass through the fully connected layer
        return out

    def train_model(self, X: np.ndarray, y: np.ndarray,
                    epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                    validation_split: float = 0.2, patience: int = 10,
                    lr_scheduler_factor: float = 0.1, lr_scheduler_patience: int = 5,
                    weight_decay: float = 1e-5, # Hardcoded default weight_decay (L2 regularization)
                    grad_clip_norm: float = 1.0, # Hardcoded default grad_clip_norm
                    training_noise_std: float = 0.05): # Added training_noise_std parameter
        """
        Trains the prediction model with a training and validation split, and early stopping.
        Now includes a learning rate scheduler, regularization (Weight Decay, Gradient Clipping),
        and noise injection during training.

        Args:
            X (np.ndarray): All input features (state vectors).
            y (np.ndarray): All target features (next day's features).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            validation_split (float): Fraction of the data to be used as validation data.
            patience (int): Number of epochs with no improvement on validation loss after which training will be stopped.
            lr_scheduler_factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
            lr_scheduler_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            weight_decay (float): L2 regularization factor for the optimizer.
            grad_clip_norm (float): Maximum norm for gradient clipping.
            training_noise_std (float): Standard deviation of Gaussian noise to add to input features during training.
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )

        # Convert numpy arrays to PyTorch tensors and move to the correct device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Create DataLoaders for batching
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay) # Added weight_decay

        # Initialize the learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor,
                                       patience=lr_scheduler_patience)

        self.train() # Set model to training mode
        print("\nStarting model training...")

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training loop
            self.train() # Ensure model is in train mode
            for batch_X, batch_y in train_loader:
                # Add Gaussian noise to the input batch_X during training
                if training_noise_std > 0:
                    noise = torch.randn_like(batch_X) * training_noise_std
                    batch_X = batch_X + noise

                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                
                optimizer.step()

            # Validation loop (NOISE IS NOT ADDED HERE - evaluate on clean data)
            self.eval() # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    val_outputs = self(batch_X_val)
                    val_loss += criterion(val_outputs, batch_y_val).item()
            val_loss /= len(val_loader) # Average validation loss over batches

            # Step the learning rate scheduler
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0: # Print first epoch and every 10th
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Optionally save the best model weights
                # torch.save(self.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
                    break # Stop training

        print("Model training complete.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X_test (np.ndarray): Input features for prediction.
                                 Shape: (num_samples, sequence_length, num_features_per_day).

        Returns:
            np.ndarray: Predicted features for the next day.
                        Shape: (num_samples, num_features_per_day).
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Convert numpy array to PyTorch tensor and move to the correct device
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            predictions = self(X_test_tensor)
        return predictions.cpu().numpy() # Move predictions back to CPU before converting to NumPy

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Evaluates the model's performance on a test set, calculating overall MSE and MAE per feature.

        Args:
            X_test (np.ndarray): Test input features.
            y_test (np.ndarray): Test target features.
            feature_names (List[str]): List of names for each feature, in the order they appear in y_test.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics (overall MSE, and MAE for each feature).
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

            predictions_scaled = self(X_test_tensor)
            mse = nn.MSELoss()(predictions_scaled, y_test_tensor).item()

            # Convert predictions and actuals back to CPU and NumPy for MAE calculation
            predictions_np = predictions_scaled.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()

            # Calculate MAE for each feature
            per_feature_mae = {}
            for i, feature_name in enumerate(feature_names):
                mae = mean_absolute_error(y_test_np[:, i], predictions_np[:, i])
                per_feature_mae[f"MAE_{feature_name}"] = mae

            print(f"Model Evaluation - Overall MSE: {mse:.4f}")
            print("Mean Absolute Error (MAE) per feature (scaled values):")
            for feature, mae_val in per_feature_mae.items():
                print(f"  {feature}: {mae_val:.4f}")

            return {"overall_mse": mse, **per_feature_mae}
        
    def save(self, path: str):
        """
        Saves model weights and architecture metadata to disk.

        Args:
            path (str): File path to save the model (e.g. 'src/models/saved_models/lstm_chronoopt.pt').
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'state_dict': self.state_dict(),
            'metadata': {
                'input_size': self.lstm.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.fc.out_features,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout.p,
            }
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> 'PredictionModel':
        """
        Loads a PredictionModel from a checkpoint file.

        Args:
            path (str): Path to the saved checkpoint.
            device (torch.device): Device to load the model onto.

        Returns:
            PredictionModel: Reconstructed model in eval mode.
        """
        checkpoint = torch.load(path, map_location=device)
        meta = checkpoint['metadata']
        model = cls(
            input_size=meta['input_size'],
            hidden_size=meta['hidden_size'],
            output_size=meta['output_size'],
            num_layers=meta['num_layers'],
            dropout_rate=meta['dropout_rate'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model

# Example usage (for internal testing of the PyTorch model)
if __name__ == "__main__":
    # Dummy data for testing the PredictionModel class
    sequence_length = 7 # X days in state
    num_features_per_day = 23 # Total numerical features per day (adjust based on DataProcessor)
    dummy_feature_names = [f"feature_{i}" for i in range(num_features_per_day)] # Dummy names

    # Create dummy data (batch_size, sequence_length, num_features_per_day)
    num_samples = 100 # Need enough samples for split
    X_dummy = np.random.rand(num_samples, sequence_length, num_features_per_day)
    y_dummy = np.random.rand(num_samples, num_features_per_day) # Predicting all features for next day

    # Initialize the model with a hardcoded dropout rate
    hidden_size = 64
    num_layers = 2
    dropout_rate = 0.2 # Hardcoded for this step
    model = PredictionModel(input_size=num_features_per_day,
                            hidden_size=hidden_size,
                            output_size=num_features_per_day,
                            num_layers=num_layers,
                            dropout_rate=dropout_rate)

    # Train the model with hardcoded regularization parameters
    model.train_model(X_dummy, y_dummy,
                      epochs=100, # Reduced epochs for dummy test
                      batch_size=16,
                      validation_split=0.2,
                      patience=10,
                      lr_scheduler_factor=0.1, # Reduce LR by 10x
                      lr_scheduler_patience=5,
                      weight_decay=1e-5, # Hardcoded for this step
                      grad_clip_norm=1.0) # Hardcoded for this step

    # Make predictions (using a subset of dummy data as test set)
    test_sample_X = np.random.rand(1, sequence_length, num_features_per_day) # Predict for one sample
    predicted_features = model.predict(test_sample_X)
    print("\nPredicted features for a test sample (first 5 values):")
    print(predicted_features[0, :5]) # Print only first 5 values for brevity

    # Evaluate (using the validation set from the split, or a separate hold-out test set if available)
    X_eval = X_dummy[80:]
    y_eval = y_dummy[80:]
    model.evaluate_model(X_eval, y_eval, dummy_feature_names)
