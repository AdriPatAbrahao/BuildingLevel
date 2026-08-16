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
import time
from config.settings import BuildingConfig

# Importa a arquitetura da rede neural e as configurações
from models.dnnmodel import SimpleNN
from config.settings import NeuralNetConfig # Importa as configurações da rede neural
from utils.artifact_contract import (
    current_artifact_contract,
    validate_artifact_contract,
)

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
        self._last_grad_stats: dict | None = None  # populated during training epochs
        self.metrics_dir: Optional[Path] = None    # set externally by caller before train()
        self.best_epoch: Optional[int] = None
        self.best_val_loss: Optional[float] = None
        self.artifact_contract: Optional[dict] = None
        print(f"NeuralNetworkManager initialized. Using device: {self.device}")

    def train(self, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, X_val_scaled: np.ndarray, y_val_scaled: np.ndarray) -> None:
        """
        Treina a rede neural nos dados JÁ NORMALIZADOS (conjuntos pré-divididos).
        """
        print("--- Preparing for NN Training with Early Stopping ---")
        print("--- NNManager: Iniciando treinamento com dados pré-normalizados ---")
        self.is_trained = False
        self.best_epoch = None
        self.best_val_loss = None
        self._validate_training_arrays(
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val_scaled,
        )

        train_loader, val_loader = self._create_dataloaders(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

        # 3. Initialize model, loss, and optimizer
        criterion, optimizer = self._initialize_model_components()

        # 4. Training loop with Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = None

        use_sched = bool(getattr(NeuralNetConfig, 'LR_SCHEDULER', False))
        if use_sched:
            sched_patience = int(getattr(NeuralNetConfig, 'LR_SCHEDULER_PATIENCE', 10))
            sched_factor = float(getattr(NeuralNetConfig, 'LR_SCHEDULER_FACTOR', 0.5))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=sched_patience, factor=sched_factor)
        else:
            scheduler = None

        print(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            epoch_start = time.time() if True else None
            train_loss = self._run_train_epoch(train_loader, criterion, optimizer)
            val_loss = self._run_eval_epoch(val_loader, criterion)
            if scheduler is not None:
                scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            epoch_dur = (time.time() - epoch_start) if epoch_start is not None else None
            if self.metrics_dir:
                try:
                    from config.settings import RunConfig
                    if getattr(RunConfig, 'METRICS_LOG_FORMAT', 'json') == 'json':
                        import json, os
                        os.makedirs(self.metrics_dir, exist_ok=True)
                        epochs_path = self.metrics_dir / 'epochs.ndjson'
                        rec = {
                            'epoch': epoch + 1,
                            'train_loss': float(train_loss),
                            'val_loss': float(val_loss),
                            'epoch_duration_sec': float(epoch_dur) if epoch_dur is not None else None,
                            'learning_rate': float(lr),
                            'gradients_mean_by_layer': (self._last_grad_stats or {}).get('mean_abs'),
                            'gradients_norm_by_layer': (self._last_grad_stats or {}).get('l2_norm')
                        }
                        with open(epochs_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early Stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                import copy as _copy
                best_model_state = _copy.deepcopy(self.model.state_dict())  # deep copy: tensors must not share storage with the live model
                print(f"  New best validation loss: {best_val_loss:.4f}. Model state saved.")
            else:
                patience_counter += 1
                print(f"  Validation loss did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs (patience {self.early_stopping_patience} reached).")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state) # Load the best model
            self.best_epoch = best_epoch
            self.best_val_loss = float(best_val_loss)
            print(
                "Loaded best model state based on validation loss "
                f"(epoch {self.best_epoch}, loss {self.best_val_loss:.6f})."
            )
        else:
            print("Warning: No best model state saved (perhaps training was too short or data was problematic).")

        self.is_trained = True
        print("--- NN Training Complete ---")
  
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

        X_scaled = np.asarray(X_scaled)
        if X_scaled.ndim != 2:
            raise ValueError(
                f"Prediction features must be a 2D array; got shape {X_scaled.shape}."
            )
        if X_scaled.shape[0] == 0:
            raise ValueError("Prediction features cannot be empty.")
        if X_scaled.shape[1] != self._input_size:
            raise ValueError(
                "Prediction feature count differs from the trained model "
                f"({X_scaled.shape[1]} != {self._input_size})."
            )
        if not np.isfinite(X_scaled).all():
            raise ValueError("Prediction features contain NaN or infinite values.")
        
        self.model.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions_normalized = self.model(X_tensor).cpu().numpy()

        return predictions_normalized

    def save_model(self, path: Path):
        """
        Salva apenas o estado do modelo e sua arquitetura.
        Não salva mais os parâmetros de normalização.
        """
        if not self.is_trained:
            raise RuntimeError("Não é permitido salvar um modelo não treinado.")
        
        contract = current_artifact_contract()
        if self._input_size != contract["input_size"]:
            raise RuntimeError("O tamanho de entrada do modelo viola o contrato atual.")
        if self._output_size != contract["output_size"]:
            raise RuntimeError("O tamanho de saída do modelo viola o contrato atual.")
        checkpoint = {
            **contract,
            'model_family': 'pytorch_mlp',
            'model_state_dict': self.model.state_dict(),
            'hidden_layers': NeuralNetConfig.HIDDEN_LAYERS,
            'dropout_rate': NeuralNetConfig.DROPOUT_RATE,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }
        try:
            torch.save(checkpoint, path)
            self.artifact_contract = contract
            print(f"--- [SAVE SUCCESS] torch.save() executado com sucesso para o caminho: '{path}' ---")
        except Exception as e:
            raise RuntimeError(f"Falha ao salvar modelo em '{path}': {e}") from e
    
    def load_model(self, path: Path) -> bool: # Recebe um objeto Path
        """
        Carrega o estado do modelo e sua arquitetura a partir de um caminho.
        Retorna True se bem-sucedido, False caso contrário.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nenhum modelo encontrado em {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            contract = validate_artifact_contract(
                checkpoint,
                artifact_label="Neural network model",
            )
            if checkpoint.get('model_family') != 'pytorch_mlp':
                raise RuntimeError("Model artifact is not a PyTorch MLP.")
            
            self._input_size = contract['input_size']
            self._output_size = contract['output_size']
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
            self.best_epoch = checkpoint.get('best_epoch')
            self.best_val_loss = checkpoint.get('best_val_loss')
            self.artifact_contract = contract
            
            self.is_trained = True
            print(f"Modelo carregado com sucesso de {path}")
            return True
        except Exception as e:
            self.is_trained = False
            self.artifact_contract = None
            raise RuntimeError(
                f"Modelo ausente, legado ou incompatível em '{path}': {e}"
            ) from e

    def run_feature_importance_analysis(
        self,
        X_val: np.ndarray,
        y_val_real: np.ndarray,
        feature_names: List[str],
        plotter,                          # ResultsPlotter instance
        *,
        feature_pipeline=None,            # FeaturePipeline for inverse transform
        classifier=None,                  # optional sklearn validity classifier
        X_val_clf: Optional[np.ndarray] = None,
        y_val_clf: Optional[np.ndarray] = None,
        n_repeats: int = 15,
        top_n: int = 25,
        run_shap: bool = True,
    ) -> None:
        """
        Compute and save Permutation Feature Importance (sklearn) and SHAP plots.

        This method is model-agnostic on the DNN side: it wraps the PyTorch
        forward pass in a simple callable so sklearn's ``permutation_importance``
        and the SHAP explainers can treat it as a black box.

        Parameters
        ----------
        X_val : np.ndarray
            Scaled validation / test features  (n_samples, n_features).
        y_val_real : np.ndarray
            Ground-truth steel values in real scale (kgf)  (n_samples,).
        feature_names : list of str
            Feature names aligned with columns of X_val.
        plotter : ResultsPlotter
            Instance whose ``output_dir`` will receive the PNG files.
        feature_pipeline : FeaturePipeline, optional
            Used to inverse-transform NN outputs to real scale before scoring.
            Without it the raw (scaled) NN output is used — PFI rankings are
            still meaningful but MSE values are in normalised space.
        classifier : sklearn estimator, optional
            Validity classifier — if provided, PFI is also run for it.
        X_val_clf : np.ndarray, optional
            Raw (unscaled) features for the classifier PFI.
            Falls back to *X_val* when None.
        y_val_clf : np.ndarray, optional
            Ground-truth 0/1 validity labels for classifier PFI.
        n_repeats : int
            Number of permutation repeats per feature (sklearn PFI).
        top_n : int
            Maximum features shown in each chart.
        run_shap : bool
            Whether to attempt SHAP analysis.  Requires ``shap`` to be
            installed; gracefully skips if the import fails.
        """
        if self.model is None or not self.is_trained:
            print("[NNManager] run_feature_importance_analysis: model not trained — skipping.")
            return

        import torch as _torch

        self.model.eval()

        # ── 1. Build a predict_fn for the NN regressor ───────────────────────
        def _predict_steel_real(X: np.ndarray) -> np.ndarray:
            """Returns predictions in real kgf scale."""
            Xt = _torch.tensor(X.astype(np.float32)).to(self.device)
            self.model.eval()
            with _torch.no_grad():
                out = self.model(Xt).cpu().numpy()
            col = out[:, 0:1] if out.ndim == 2 else out.reshape(-1, 1)
            if feature_pipeline is not None and hasattr(
                feature_pipeline, "inverse_transform_outputs"
            ):
                return feature_pipeline.inverse_transform_outputs(col)[:, 0]
            return col[:, 0]

        # ── 2. Permutation Feature Importance — DNN regressor ────────────────
        print("\n[NNManager] Running sklearn PFI for DNN regressor…")
        try:
            plotter.plot_permutation_importance_sklearn(
                predict_fn=_predict_steel_real,
                X_val=X_val,
                y_val=y_val_real,
                feature_names=feature_names,
                task="regression",
                n_repeats=n_repeats,
                top_n=top_n,
                output_file="pfi_sklearn_steel_regression.png",
            )
        except Exception as exc:
            print(f"[NNManager] PFI (regression) failed: {exc}")

        # ── 3. Permutation Feature Importance — validity classifier ──────────
        _Xc = X_val_clf if X_val_clf is not None else X_val
        if classifier is not None and y_val_clf is not None:
            print("\n[NNManager] Running sklearn PFI for validity classifier…")
            try:
                classes = list(getattr(classifier, "classes_", [0, 1]))
                idx_pos = classes.index(1) if 1 in classes else 1

                def _predict_clf(X: np.ndarray) -> np.ndarray:
                    if hasattr(classifier, "predict_proba"):
                        return classifier.predict_proba(X)[:, idx_pos]
                    return classifier.predict(X).astype(float)

                plotter.plot_permutation_importance_sklearn(
                    predict_fn=_predict_clf,
                    X_val=_Xc,
                    y_val=np.asarray(y_val_clf, dtype=float),
                    feature_names=feature_names,
                    task="classification",
                    n_repeats=n_repeats,
                    top_n=top_n,
                    output_file="pfi_sklearn_validity_classifier.png",
                )
            except Exception as exc:
                print(f"[NNManager] PFI (classification) failed: {exc}")

        # ── 4. SHAP analysis ─────────────────────────────────────────────────
        if run_shap:
            print("\n[NNManager] Running SHAP analysis for DNN regressor…")
            try:
                plotter.plot_shap_summary(
                    model=self.model,
                    X_background=X_val,
                    X_explain=X_val,
                    feature_names=feature_names,
                    feature_pipeline=feature_pipeline,
                    output_file="shap_summary_steel.png",
                )
            except Exception as exc:
                print(f"[NNManager] SHAP analysis failed: {exc}")
            finally:
                # Ensure model returns to the configured device after CPU move
                self.model.to(self.device)

        print("[NNManager] Feature importance analysis complete.")

    def _validate_training_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Validate the complete train/validation contract before model creation."""
        arrays = {
            "X_train": np.asarray(X_train),
            "y_train": np.asarray(y_train),
            "X_val": np.asarray(X_val),
            "y_val": np.asarray(y_val),
        }
        for name, array in arrays.items():
            if array.ndim != 2:
                raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
            if array.shape[0] == 0:
                raise ValueError(f"{name} cannot be empty.")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinite values.")

        if arrays["X_train"].shape[0] != arrays["y_train"].shape[0]:
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if arrays["X_val"].shape[0] != arrays["y_val"].shape[0]:
            raise ValueError("X_val and y_val must contain the same number of rows.")
        if arrays["X_train"].shape[1] != arrays["X_val"].shape[1]:
            raise ValueError("Training and validation feature counts must match.")
        if arrays["y_train"].shape[1] != arrays["y_val"].shape[1]:
            raise ValueError("Training and validation output counts must match.")
        if arrays["X_train"].shape[1] != self._input_size:
            raise ValueError(
                "Training feature count differs from NeuralNetConfig.INPUT_SIZE "
                f"({arrays['X_train'].shape[1]} != {self._input_size})."
            )
        if arrays["y_train"].shape[1] != self._output_size:
            raise ValueError(
                "Training output count differs from NeuralNetConfig.OUTPUT_SIZE "
                f"({arrays['y_train'].shape[1]} != {self._output_size})."
            )
        if arrays["X_train"].shape[0] < 2:
            raise ValueError(
                "At least two training samples are required because the model uses BatchNorm1d."
            )

    def _safe_train_batch_size(self, num_samples: int) -> int:
        """Choose a batch size that never leaves a one-sample BatchNorm batch."""
        if num_samples < 2:
            raise ValueError("At least two training samples are required.")
        if self.batch_size < 2:
            raise ValueError("NeuralNetConfig.BATCH_SIZE must be at least 2.")

        candidate = min(int(self.batch_size), int(num_samples))
        while candidate >= 2:
            if num_samples % candidate != 1:
                return candidate
            candidate -= 1
        return int(num_samples)

    def _create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_batch_size = self._safe_train_batch_size(len(train_dataset))
        from config.settings import RunConfig
        generator = torch.Generator()
        generator.manual_seed(int(getattr(RunConfig, "SEED", 42)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=generator,
        )

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
        loss_type = getattr(NeuralNetConfig, 'LOSS_TYPE', 'mse')
        if loss_type == 'huber':
            criterion = nn.HuberLoss()
        else:
            criterion = nn.MSELoss()
        weight_decay = float(getattr(NeuralNetConfig, 'WEIGHT_DECAY', 0.0))
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        return criterion, optimizer

    def _run_train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Executa uma época de treino."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        last_stats_mean = {}
        last_stats_norm = {}
        for i, (batch_X, batch_y) in enumerate(loader):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if i == len(loader) - 1:
                try:
                    from config.settings import RunConfig
                    if getattr(RunConfig, 'LOG_EPOCH_GRADIENTS', 'last_batch') == 'last_batch':
                        for name, p in self.model.named_parameters():
                            if p.grad is None:
                                continue
                            g = p.grad.detach()
                            last_stats_mean[name] = float(g.abs().mean().item())
                            last_stats_norm[name] = float(g.norm(2).item())
                except Exception:
                    pass
            optimizer.step()
            batch_samples = int(batch_X.shape[0])
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples
        self._last_grad_stats = {'mean_abs': last_stats_mean, 'l2_norm': last_stats_norm}
        if total_samples == 0:
            raise ValueError("Training DataLoader produced no samples.")
        return total_loss / total_samples

    def _run_eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Executa uma época de avaliação (validação)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                batch_samples = int(batch_X.shape[0])
                total_loss += loss.item() * batch_samples
                total_samples += batch_samples
        if total_samples == 0:
            raise ValueError("Validation DataLoader produced no samples.")
        return total_loss / total_samples
