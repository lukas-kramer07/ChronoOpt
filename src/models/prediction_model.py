# src/models/prediction_model.py
# This file will contain the PyTorch-based prediction model (e.g., LSTM)
# that predicts the next day's full set of features, with GPU support.

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# 1. Device Configuration: Check for CUDA (GPU) availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PredictionModel(nn.Module):
    """
    A PyTorch-based prediction model (e.g., LSTM) that predicts the next day's
    full set of features based on a sequence of past daily features.
    Supports GPU acceleration if available.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        Initializes the PredictionModel.

        Args:
            input_size (int): The number of features per day in the input sequence.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            output_size (int): The number of features to predict for the next day.
                               This should be equal to input_size.
            num_layers (int): Number of recurrent layers.
        """
        super(PredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Move the entire model to the selected device (CPU or GPU)
        self.to(self.device)

        print(f"PredictionModel initialized on {self.device} with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}")

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
        out = self.fc(out[:, -1, :]) # Get output from the last time step
        return out

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Trains the prediction model.

        Args:
            X_train (np.ndarray): Training input features (state vectors).
                                  Shape: (num_samples, sequence_length, num_features_per_day).
            y_train (np.ndarray): Training target features (next day's features).
                                  Shape: (num_samples, num_features_per_day).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
        """
        # Convert numpy arrays to PyTorch tensors and move to the correct device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # Create DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.train() # Set model to training mode
        print("\nStarting model training...")
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                # batch_X and batch_y are already on the correct device due to DataLoader
                # created from tensors moved to device

                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
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

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model's performance on a test set.

        Args:
            X_test (np.ndarray): Test input features.
            y_test (np.ndarray): Test target features.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics (e.g., MSE).
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

            predictions = self(X_test_tensor)
            mse = nn.MSELoss()(predictions, y_test_tensor).item()
            print(f"Model Evaluation - MSE: {mse:.4f}")
            return {"mse": mse}

# Example usage (for internal testing of the PyTorch model)
if __name__ == "__main__":
    # Dummy data for testing the PredictionModel class
    sequence_length = 7 # X days in state
    num_features_per_day = 23 # Total numerical features per day (adjust based on DataProcessor)

    # Create dummy data (batch_size, sequence_length, num_features_per_day)
    # and corresponding targets (batch_size, num_features_per_day)
    num_samples = 100
    X_dummy = np.random.rand(num_samples, sequence_length, num_features_per_day)
    y_dummy = np.random.rand(num_samples, num_features_per_day) # Predicting all features for next day

    # Initialize the model
    hidden_size = 64
    num_layers = 2
    model = PredictionModel(input_size=num_features_per_day,
                            hidden_size=hidden_size,
                            output_size=num_features_per_day,
                            num_layers=num_layers)

    # Train the model
    model.train_model(X_dummy, y_dummy, epochs=50, batch_size=16)

    # Make predictions
    test_sample_X = np.random.rand(1, sequence_length, num_features_per_day) # Predict for one sample
    predicted_features = model.predict(test_sample_X)
    print("\nPredicted features for a test sample (first 5 values):")
    print(predicted_features[0, :5]) # Print only first 5 values for brevity

    # Evaluate (using a subset of dummy data as test set)
    X_eval = X_dummy[80:]
    y_eval = y_dummy[80:]
    model.evaluate_model(X_eval, y_eval)
