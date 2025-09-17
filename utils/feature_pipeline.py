import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import joblib # Usaremos joblib para salvar/carregar os scalers de forma robusta

from geometry.length_input_processor import LengthProcessor
from utils.feature_engineer import FeatureEngineer

class FeaturePipeline:
    """
    Encapsula todo o processo de transformação de dados, desde o arquivo CSV
    até o vetor de features normalizado, garantindo consistência entre
    treinamento e inferência.
    """
    def __init__(self):
        self.input_processor = LengthProcessor() # Assume LengthProcessor por padrão
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False # Flag para saber se os scalers foram treinados

    def fit_transform(self, feature_vectors: List[List[float]], outputs: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Treina os scalers e transforma os dados. Usado durante o treinamento.
        """
        X_np = np.array(feature_vectors, dtype=np.float32)
        y_np = np.array(outputs, dtype=np.float32)

        X_scaled = self.scaler_X.fit_transform(X_np)
        y_scaled = self.scaler_y.fit_transform(y_np)
        
        self.is_fitted = True
        print("FeaturePipeline foi 'fitado' (treinado).")
        
        return X_scaled, y_scaled

    def transform_features(self, feature_vectors: List[List[float]]) -> np.ndarray:
        """
        Apenas transforma os features de entrada usando o scaler já treinado.
        Usado durante a inferência.
        """
        if not self.is_fitted:
            raise RuntimeError("A pipeline precisa ser 'fitada' antes de transformar dados.")
        
        X_np = np.array(feature_vectors, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X_np)
        return X_scaled

    def inverse_transform_outputs(self, predictions_scaled: np.ndarray) -> np.ndarray:
        """
        Converte as predições normalizadas de volta para a escala original.
        Usado durante a inferência.
        """
        if not self.is_fitted:
            raise RuntimeError("A pipeline precisa ser 'fitada' antes de transformar dados.")
            
        return self.scaler_y.inverse_transform(predictions_scaled)

    def process_csv_to_features(self, csv_path: str) -> List[float]:
        """
        Executa o fluxo completo de um CSV para um vetor de features.
        Esta é a função principal para a inferência.
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
        """Salva os scalers treinados em um arquivo."""
        if not self.is_fitted:
            print("Aviso: Tentando salvar uma pipeline não treinada.")
            return
        joblib.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, path)
        print(f"Pipeline (scalers) salva em {path}")

    def load(self, path: str = "feature_pipeline.pkl"):
        """Carrega os scalers de um arquivo."""
        try:
            scalers = joblib.load(path)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
            self.is_fitted = True
            print(f"Pipeline (scalers) carregada de {path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo da pipeline não encontrado em {path}. Treine o modelo primeiro.")
