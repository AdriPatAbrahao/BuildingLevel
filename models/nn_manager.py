import numpy as np
from typing import List, Optional, Dict, Any # Added Dict, Any

# Assuming dnnmodel is in algorithm directory relative to project root
from models.dnnmodel import train_model, SimpleNN, test_model

class NeuralNetworkManager:
    """
    Manages the lifecycle of the Neural Network model, including training,
    storing normalization parameters, and making predictions.
    """

    def __init__(self, input_size: Optional[int] = None, hidden_size: Optional[int] = None, output_size: Optional[int] = None):
        """
        Initializes the Neural Network Manager.

        Args:
            input_size (Optional[int]): Specify input size, otherwise determined from data during train.
            hidden_size (Optional[int]): Specify hidden size, otherwise uses a default.
            output_size (Optional[int]): Specify output size, otherwise uses a default.
        """
        self.model: Optional[SimpleNN] = None
        # Store normalization parameters calculated during training
        self.norm_params: Optional[Dict[str, np.ndarray]] = None

        # Store config if provided, otherwise they get set in train
        self._input_size = input_size
        self._hidden_size = hidden_size if hidden_size else 128 # Default hidden size
        self._output_size = output_size if output_size else 2 # Default output size (e.g., steel, concrete)


    def train(self, feature_vectors: List[List[float]], outputs: List[List[float]],
              num_epochs: int = 100, batch_size: int = 16, learning_rate: float = 0.001) -> Dict[str, np.ndarray]:
        """
        Trains the neural network on the provided data and stores normalization parameters.

        Args:
            feature_vectors (List[List[float]]): List of input feature vectors (e.g., lengths or binary flags).
            outputs (List[List[float]]): List of corresponding target output vectors (e.g., [steel, concrete]).
            num_epochs (int): Max number of epochs for training.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
             Dict[str, np.ndarray]: The normalization parameters used during training.
                                    Returns None if training fails.
        """
        print("--- Preparing for NN Training ---")
        if not feature_vectors or not outputs:
             raise ValueError("Training failed: feature_vectors or outputs list is empty.")

        # Determine input size from data if not set previously
        if self._input_size is None:
             self._input_size = len(feature_vectors[0])
             print(f"Input size determined from data: {self._input_size}")
        # Ensure output size matches data if possible
        if self._output_size != len(outputs[0]):
             print(f"Warning: Output size mismatch. Manager expected {self._output_size}, data has {len(outputs[0])}. Using data's size.")
             self._output_size = len(outputs[0])

        # Create the model instance
        print(f"Creating SimpleNN model: Input={self._input_size}, Hidden={self._hidden_size}, Output={self._output_size}")
        self.model = SimpleNN(input_size=self._input_size, hidden_size=self._hidden_size, output_size=self._output_size)

        # Convert lists to numpy arrays for train_model
        X_train_np = np.array(feature_vectors, dtype=np.float32)
        y_train_np = np.array(outputs, dtype=np.float32)

        # Call the training function and store normalization parameters
        try:
            print("Calling train_model function...")
            self.norm_params = train_model(
                self.model, X_train_np, y_train_np,
                num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
            )
            print("--- NN Training Complete ---")
            return self.norm_params # Return the params
        except Exception as e:
            print(f"Error during model training: {e}")
            # import traceback
            # print(traceback.format_exc())
            self.model = None # Ensure model is None if training failed
            self.norm_params = None
            raise # Re-raise the exception
        
    def predict(self, feature_vectors: List[List[float]]) -> np.ndarray:
        """
        Makes predictions using the trained model and applies denormalization.

        Args:
            feature_vectors (List[List[float]]): List of input feature vectors for prediction.

        Returns:
            np.ndarray: Array of denormalized predictions [steel, concrete].
                        Returns an empty array if prediction is not possible.
        """
        print("--- Making Predictions ---")
        if self.model is None:
            raise RuntimeError("Prediction failed: Model has not been trained yet. Call train() first.")
        if self.norm_params is None:
             # Option 1: Raise error - safer if normalization is critical
             raise RuntimeError("Prediction failed: Normalization parameters are missing. Was the model trained correctly?")
             # Option 2: Predict without normalization (less safe, likely inaccurate)
             # print("Warning: Normalization parameters not found. Predicting on raw data.")
             # X_test_np = np.array(feature_vectors, dtype=np.float32)
             # with torch.no_grad():
             #    inputs_tensor = torch.tensor(X_test_np, dtype=torch.float32)
             #    raw_predictions = self.model(inputs_tensor).numpy()
             # return raw_predictions # Return raw predictions in this case

        if not feature_vectors:
            print("Warning: Input feature_vectors for prediction is empty.")
            return np.array([])

        X_test_np = np.array(feature_vectors, dtype=np.float32)

        # Call test_model, passing the required normalization parameters
        try:
            print("Calling test_model (applies input normalization)...")
            # test_model applies X normalization and returns predictions on y's normalized scale
            predictions_normalized = test_model(self.model, X_test_np, self.norm_params)

            if predictions_normalized.size == 0:
                 print("Warning: test_model returned empty predictions.")
                 return np.array([])

            # Denormalize the predictions using stored y_mean and y_std
            print("Denormalizing predictions...")
            y_mean = self.norm_params['y_mean']
            y_std_safe = self.norm_params['y_std'] # Use the safe std dev

            # Denormalization: prediction = prediction_normalized * std + mean
            predictions_denormalized = predictions_normalized * y_std_safe + y_mean
            print("--- Prediction Complete ---")

            return predictions_denormalized

        except Exception as e:
             print(f"Error during model prediction: {e}")
             # import traceback
             # print(traceback.format_exc())
             return np.array([]) # Return empty array on error


    

