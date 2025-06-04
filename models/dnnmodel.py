import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any

class SimpleNN(nn.Module):
    """
    Defines a simple feed-forward neural network architecture.

    Uses Linear layers, BatchNorm1d for stabilization, ReLU activation,
    and Dropout for regularization.
    """

    def __init__(self, input_size=24, hidden_size=128, output_size=2):
        """
        Initializes the neural network layers.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layers.
            output_size (int): Number of output values to predict.
        """
        super(SimpleNN, self).__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive integers.")
        self.input_size = input_size

        # Deeper network with more capacity
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
   
        # Forward pass logic
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

# --- Function to train the model ---
def train_model(model: SimpleNN, X_train: np.ndarray, y_train: np.ndarray,
                num_epochs: int = 100, batch_size: int = 16, learning_rate: float = 0.001,
                early_stopping_patience: int = 20) -> Dict[str, np.ndarray]:
    """
    Trains the neural network model using the provided data.

    Performs Z-score normalization on input features (X) and target outputs (y)
    based *only* on the training data.

    Args:
        model (SimpleNN): The neural network model instance to train.
        X_train (np.ndarray): Training input features (samples, num_features).
        y_train (np.ndarray): Training target outputs (samples, num_outputs).
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Learning rate for the optimizer.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the normalization parameters:
                                 {'X_mean': ..., 'X_std': ..., 'y_mean': ..., 'y_std': ...}
    """
    # --- Input Validation ---
    if not isinstance(X_train, np.ndarray): X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray): y_train = np.array(y_train)
    
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data (X_train or y_train) cannot be empty.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X_train has {X_train.shape[0]}, y_train has {y_train.shape[0]}.")
    if X_train.shape[1] != model.input_size:
        raise ValueError(f"Input feature dimension mismatch: X_train has {X_train.shape[1]} features, model expects {model.input_size}.")
    if y_train.shape[1] != model.fc4.out_features:
        raise ValueError(f"Output dimension mismatch: y_train has {y_train.shape[1]} outputs, model expects {model.fc4.out_features}.")

    print(f"Starting training with {X_train.shape[0]} samples...")

    # --- Normalization ---
    print("Calculating normalization parameters from training data...")
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    # Add small epsilon to prevent division by zero for features with no variance
    X_std_safe = np.where(X_std == 0, 1e-8, X_std)

    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_std_safe = np.where(y_std == 0, 1e-8, y_std) # Handle zero variance in outputs too

    print(f"X_mean: {X_mean}")
    print(f"X_std: {X_std}")
    print(f"y_mean: {y_mean}")
    print(f"y_std: {y_std}")


    X_train_norm = (X_train - X_mean) / X_std_safe
    y_train_norm = (y_train - y_mean) / y_std_safe
    print("Training data normalized.")

    # --- Training Setup ---
    criterion = nn.MSELoss() # Mean Squared Error loss for regression
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # AdamW is often a good default
    # Learning rate scheduler: reduces LR if loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=early_stopping_patience  // 2 if early_stopping_patience > 4 else 5) # Reduce LR scheduler patience relative to early stopping

    # --- Training Loop ---
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train() # Set model to training mode (enables dropout/batchnorm updates)
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data each epoch
        indices = np.random.permutation(len(X_train_norm))
        X_train_shuffled = X_train_norm[indices]
        y_train_shuffled = y_train_norm[indices]

        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i : i + batch_size]
            batch_y = y_train_shuffled[i : i + batch_size]

            # Convert batch to PyTorch tensors
            inputs_tensor = torch.tensor(batch_X, dtype=torch.float32)
            targets_tensor = torch.tensor(batch_y, dtype=torch.float32)

            # Forward pass
            optimizer.zero_grad() # Clear previous gradients
            outputs_tensor = model(inputs_tensor)
            loss = criterion(outputs_tensor, targets_tensor)

            # Backward pass and optimization
            loss.backward() # Calculate gradients
            optimizer.step() # Update model weights

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)

        # Early Stopping Check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # Optional: Save the best model state here
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0: # Print progress every 10 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1} due to no improvement for {early_stopping_patience} epochs.")
            break

    print(f"Training finished. Best validation loss: {best_loss:.6f}")

    # Return the calculated normalization parameters
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std_safe, # Return the safe version
        'y_mean': y_mean,
        'y_std': y_std_safe # Return the safe version
    }
    return norm_params


# --- Function to test the model (make predictions) ---
def test_model(model: SimpleNN, X_test: np.ndarray, norm_params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Tests the neural network model on new data using stored normalization parameters.

    Args:
        model (SimpleNN): The trained neural network model instance.
        X_test (np.ndarray): The input features for testing (samples, num_features).
        norm_params (Dict[str, np.ndarray]): Dictionary containing 'X_mean' and 'X_std'
                                            calculated during training.

    Returns:
        np.ndarray: The raw predictions from the model (on the normalized scale of y).
                    Shape: (samples, num_outputs).
    """
    if not isinstance(X_test, np.ndarray): X_test = np.array(X_test)
    if X_test.size == 0:
        print("Warning: X_test is empty. Returning empty array.")
        return np.array([])
    if 'X_mean' not in norm_params or 'X_std' not in norm_params:
         raise ValueError("Normalization parameters ('X_mean', 'X_std') not found in norm_params dict.")

    X_mean = norm_params['X_mean']
    X_std_safe = norm_params['X_std'] # Use the safe std dev

    if X_test.shape[1] != len(X_mean):
         raise ValueError(f"Input feature dimension mismatch: X_test has {X_test.shape[1]} features, expected {len(X_mean)} based on training normalization.")

    # --- Normalize the test data using TRAINING parameters ---
    X_test_norm = (X_test - X_mean) / X_std_safe
    print(f"Test data normalized using training parameters.")

    # --- Prediction ---
    model.eval() # Set model to evaluation mode (disables dropout, uses running averages for batchnorm)
    with torch.no_grad(): # Disable gradient calculations for inference
        inputs_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
        predictions_tensor = model(inputs_tensor)

    # Return the raw numpy array output from the model
    # Note: These predictions are still on the NORMALIZED scale of y
    return predictions_tensor.numpy()