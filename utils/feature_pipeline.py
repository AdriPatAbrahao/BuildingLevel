import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import joblib # Usaremos joblib para salvar/carregar os scalers de forma robusta

from geometry.length_input_processor import LengthProcessor
from utils.feature_engineer import FeatureEngineer

class FeaturePipeline:
    """
    End-to-end feature transformation pipeline for training and inference.

    Reads geometry from CSV and builds feature vectors, applying `StandardScaler`
    consistently between training and inference. Scalers are persisted with
    `joblib` for reproducibility.
    """
    def __init__(self, input_processor=None):
        self.input_processor = input_processor if input_processor is not None else LengthProcessor()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False # Flag para saber se os scalers foram treinados

    def fit_transform(self, feature_vectors: List[List[float]], outputs: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform features and outputs (training phase).

        Parameters
        ----------
        feature_vectors : List[List[float]]
            Raw feature vectors extracted from geometry.
        outputs : List[List[float]]
            Target outputs (e.g., steel and concrete if multi-output).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Scaled features and scaled outputs.
        """
        X_np = np.array(feature_vectors, dtype=np.float32)
        y_np = np.array(outputs, dtype=np.float32)

        X_scaled = self.scaler_X.fit_transform(X_np)
        y_scaled = self.scaler_y.fit_transform(y_np)
        
        self.is_fitted = True
        print("FeaturePipeline foi 'fitado' (treinado).")
        
        return X_scaled, y_scaled

    def fit(self, feature_vectors: List[List[float]], outputs: List[List[float]]) -> None:
        X_np = np.array(feature_vectors, dtype=np.float32)
        y_np = np.array(outputs, dtype=np.float32)
        self.scaler_X.fit(X_np)
        self.scaler_y.fit(y_np)
        self.is_fitted = True

    def transform_outputs(self, outputs: List[List[float]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("A pipeline precisa ser 'fitada' antes de transformar dados.")
        y_np = np.array(outputs, dtype=np.float32)
        return self.scaler_y.transform(y_np)

    def transform_features(self, feature_vectors: List[List[float]]) -> np.ndarray:
        """
        Transform features using the fitted scaler (inference phase).

        Parameters
        ----------
        feature_vectors : List[List[float]]
            Raw feature vectors.

        Returns
        -------
        np.ndarray
            Scaled features.
        """
        if not self.is_fitted:
            raise RuntimeError("A pipeline precisa ser 'fitada' antes de transformar dados.")
        
        X_np = np.array(feature_vectors, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X_np)
        return X_scaled

    def inverse_transform_outputs(self, predictions_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale.

        Parameters
        ----------
        predictions_scaled : np.ndarray
            Model outputs in scaled units.

        Returns
        -------
        np.ndarray
            Descaled predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("A pipeline precisa ser 'fitada' antes de transformar dados.")
            
        return self.scaler_y.inverse_transform(predictions_scaled)

    def process_csv_to_features(self, csv_path: str) -> List[float]:
        """
        Full flow from CSV to feature vector (inference helper).

        Parameters
        ----------
        csv_path : str
            Path to the input CSV file.

        Returns
        -------
        List[float]
            Extracted feature vector.
        """
        # Define o caminho do CSV no processador de input
        self.input_processor.csv_path = csv_path
        
        # 1. Lê segmentos do CSV
        segments = self.input_processor.read_length_from_csv()
        if not segments:
            raise ValueError(f"Não foi possível ler segmentos do arquivo '{csv_path}'")

        # 2. Processa segmentos para obter geometria
        column_polygons, beam_definitions = self.input_processor.process_segments(segments)

        # 3. Extrai o vetor de features
        feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
        feature_vector = feature_engineer.extract_features()
        
        return feature_vector

    def save(self, path: str = "feature_pipeline.pkl"):
        """Persist fitted scalers to a file using `joblib`."""
        if not self.is_fitted:
            print("Aviso: Tentando salvar uma pipeline não treinada.")
            return
        joblib.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, path)
        print(f"Pipeline (scalers) salva em {path}")

    def load(self, path: str = "feature_pipeline.pkl"):
        """Load scalers from file and mark the pipeline as fitted."""
        try:
            scalers = joblib.load(path)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
            self.is_fitted = True
            print(f"Pipeline (scalers) carregada de {path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo da pipeline não encontrado em {path}. Treine o modelo primeiro.")
