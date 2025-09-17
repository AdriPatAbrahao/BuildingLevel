from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Any, Tuple
import os # Para salvar/carregar o modelo
from config.settings import BuildingConfig

# Importa a arquitetura da rede neural e as configurações
from models.dnnmodel import SimpleNN
from config.settings import NeuralNetConfig # Importa as configurações da rede neural

class NeuralNetworkManager:
    """
    Manages the lifecycle of the Neural Network model.
    Esta versão assume que os dados de entrada (features e outputs)
    JÁ ESTÃO NORMALIZADOS por uma pipeline externa.
    """

    def __init__(self):

        self.model: Optional[SimpleNN] = None
        self.is_trained: bool = False # Flag para indicar se o modelo foi treinado

        # Carrega configurações do NeuralNetConfig
        self._input_size = NeuralNetConfig.INPUT_SIZE
        self._output_size = NeuralNetConfig.OUTPUT_SIZE
        self.learning_rate = NeuralNetConfig.LEARNING_RATE
        self.num_epochs = NeuralNetConfig.NUM_EPOCHS
        self.batch_size = NeuralNetConfig.BATCH_SIZE
        self.early_stopping_patience = NeuralNetConfig.EARLY_STOPPING_PATIENCE
        self.validation_split_ratio = NeuralNetConfig.VALIDATION_SPLIT_RATIO

        # Configura o dispositivo (GPU se disponível, CPU caso contrário)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NeuralNetworkManager initialized. Using device: {self.device}")

    def train(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Treina a rede neural nos dados JÁ NORMALIZADOS.
        Retorna os dados de teste (também normalizados) para avaliação.
        """
        print("--- Preparing for NN Training with Early Stopping ---")
        print("--- NNManager: Iniciando treinamento com dados pré-normalizados ---")
        if X_scaled.size == 0 or y_scaled.size == 0:
            raise ValueError("Treinamento falhou: os arrays de dados normalizados estão vazios.")

        # Valida e define os tamanhos de entrada/saída dinamicamente
        self._validate_and_set_sizes(X_scaled, y_scaled)

                # 1. Dividir dados e criar DataLoaders
        #    A divisão agora é mais simples, pois não há dados "originais" para guardar.
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y_scaled, test_size=NeuralNetConfig.TEST_SPLIT_RATIO, random_state=42
        )
        train_loader, val_loader = self._create_dataloaders(X_train_val, y_train_val)

        # 3. Initialize model, loss, and optimizer
        criterion, optimizer = self._initialize_model_components()

        # 4. Training loop with Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            train_loss = self._run_train_epoch(train_loader, criterion, optimizer)
            val_loss = self._run_eval_epoch(val_loader, criterion)
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early Stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict() # Save the best model state
                print(f"  New best validation loss: {best_val_loss:.4f}. Model state saved.")
            else:
                patience_counter += 1
                print(f"  Validation loss did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs (patience {self.early_stopping_patience} reached).")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state) # Load the best model
            print("Loaded best model state based on validation loss.")
        else:
            print("Warning: No best model state saved (perhaps training was too short or data was problematic).")

        self.is_trained = True
        print("--- NN Training Complete ---")
        return X_test, y_test
  
    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model and applies denormalization.

        Args:
            feature_vectors (List[List[float]]): List of input feature vectors for prediction.

        Returns:
            np.ndarray: Array of denormalized predictions [steel, concrete].
                        Returns an empty array if prediction is not possible.
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Prediction failed: Model has not been trained yet or training failed. Call train() first.")
        
        self.model.eval() # Garante que o modelo está em modo de avaliação
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_normalized = self.model(X_tensor).cpu().numpy()

        return predictions_normalized

    def save_model(self, path: Path):
        """
        Salva apenas o estado do modelo e sua arquitetura.
        Não salva mais os parâmetros de normalização.
        """
        if not self.is_trained:
            print("Aviso: Modelo não treinado. Nada para salvar.")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self._input_size,
            'output_size': self._output_size,
            'hidden_layers': NeuralNetConfig.HIDDEN_LAYERS,
            'dropout_rate': NeuralNetConfig.DROPOUT_RATE
        }
        try:
            torch.save(checkpoint, path)
            print(f"--- [SAVE SUCCESS] torch.save() executado com sucesso para o caminho: '{path}' ---")
        except Exception as e:
            print(f"--- [SAVE FAILED] torch.save() FALHOU com um erro: {e} ---")
    
    def load_model(self, path: Path) -> bool: # Recebe um objeto Path
        """
        Carrega o estado do modelo e sua arquitetura a partir de um caminho.
        Retorna True se bem-sucedido, False caso contrário.
        """
        if not os.path.exists(path):
            print(f"Erro: Nenhum modelo encontrado em {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self._input_size = checkpoint['input_size']
            self._output_size = checkpoint['output_size']
            hidden_layers = checkpoint['hidden_layers']
            dropout_rate = checkpoint['dropout_rate']

            self.model = SimpleNN(
                input_size=self._input_size, 
                output_size=self._output_size, 
                hidden_layers=hidden_layers, 
                dropout_rate=dropout_rate
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_trained = True
            print(f"Modelo carregado com sucesso de {path}")
            return True
        except Exception as e:
            print(f"Erro ao carregar o modelo de {path}: {e}")
            self.is_trained = False
            return False

    def _validate_and_set_sizes(self, X: np.ndarray, y: np.ndarray):
        """Valida e define os tamanhos de entrada/saída a partir dos dados."""
        if self._input_size != X.shape[1]:
            print(f"Aviso: INPUT_SIZE da config ({self._input_size}) é diferente do dado ({X.shape[1]}). Usando o tamanho do dado.")
            self._input_size = X.shape[1]
        if self._output_size != y.shape[1]:
            print(f"Aviso: OUTPUT_SIZE da config ({self._output_size}) é diferente do dado ({y.shape[1]}). Usando o tamanho do dado.")
            self._output_size = y.shape[1]

    def _create_dataloaders(self, X_train_val: np.ndarray, y_train_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Cria DataLoaders de treino e validação a partir de um conjunto de dados."""
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=NeuralNetConfig.VALIDATION_SPLIT_RATIO, random_state=42
        )

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def _initialize_model_components(self) -> Tuple[nn.Module, optim.Optimizer]:
        """Inicializa o modelo, a função de perda e o otimizador."""
        self.model = SimpleNN(
            input_size=self._input_size, 
            output_size=self._output_size,
            hidden_layers=NeuralNetConfig.HIDDEN_LAYERS,
            dropout_rate=NeuralNetConfig.DROPOUT_RATE
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return criterion, optimizer

    def _run_train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Executa uma época de treino."""
        self.model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _run_eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Executa uma época de avaliação (validação)."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(loader)