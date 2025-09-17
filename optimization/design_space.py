# Em optimization/design_space.py

import pandas as pd
import numpy as np
from config import paths  # Importa seus caminhos configurados

class DesignSpace:
    """
    Define o espaço de busca para a otimização a partir de um arquivo CSV semente.
    
    Lê o arquivo CSV que contém as coordenadas, vetores de direção e comprimentos
    máximos para cada variável de projeto (segmento de pilar).
    """
    def __init__(self, seed_csv_path=paths.SEED_VECTOR_CSV_OPTMIZATION):
        """
        Inicializa e carrega os dados do espaço de busca.

        Args:
            seed_csv_path (Path): Caminho para o arquivo CSV semente.
        """
        print(f"--- Inicializando Design Space a partir de '{seed_csv_path.name}' ---")
        try:
            self.seed_df = pd.read_csv(seed_csv_path, delimiter=';')
            self._validate_csv()
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo CSV semente não encontrado em: {seed_csv_path}")
        except Exception as e:
            raise ValueError(f"Erro ao ler ou validar o CSV semente: {e}")

        # Extrai os limites e informações importantes
        self.num_variables = len(self.seed_df)
        initial_guess_from_csv = self.seed_df['length'].astype(float).values
        # Limite inferior: o próprio comprimento inicial definido no CSV
        self.lower_bounds = initial_guess_from_csv.copy()
        self.upper_bounds = self.seed_df['maxlength'].astype(float).values
        # Usa exatamente os comprimentos do CSV como chute inicial
        self.initial_guess = initial_guess_from_csv.copy()

        print(f"   - Espaço de busca definido com {self.num_variables} variáveis.")
        print(f"   - Limites inferiores (min=length inicial): {self.lower_bounds}")
        print(f"   - Limites superiores (maxlength): {self.upper_bounds}")
        print(f"   - Chute inicial (length): {self.initial_guess}")
        print("--- Design Space pronto ---")

    def _validate_csv(self):
        """Verifica se o DataFrame carregado contém as colunas necessárias."""
        required_columns = ['x', 'y', 'dx', 'dy', 'length', 'maxlength']
        if not all(col in self.seed_df.columns for col in required_columns):
            raise ValueError(f"O CSV semente deve conter as colunas: {required_columns}")

    def get_bounds(self) -> list:
        """
        Retorna os limites no formato esperado pela função `scipy.optimize.differential_evolution`.
        
        Returns:
            list: Uma lista de tuplas, onde cada tupla é (min_bound, max_bound).
        """
        return list(zip(self.lower_bounds, self.upper_bounds))

    def create_geometry_from_vector(self, vector: np.ndarray) -> pd.DataFrame:
        """
        Cria um novo DataFrame de geometria usando um vetor de comprimentos fornecido.
        
        Este método é crucial, pois traduz um vetor de decisão do otimizador
        (ex: [150.5, 200.1, ...]) de volta para um formato de geometria completo
        que o `LengthProcessor` pode entender.

        Args:
            vector (np.ndarray): O vetor de comprimentos (a solução candidata).

        Returns:
            pd.DataFrame: Um DataFrame com a geometria completa (start/end points).
        """
        if len(vector) != self.num_variables:
            raise ValueError(f"Vetor de entrada tem {len(vector)} elementos, mas o esperado era {self.num_variables}.")

        new_df = self.seed_df.copy()
        new_df['length'] = vector

        new_df['end_x'] = new_df['x'] + new_df['dx'] * new_df['length']
        new_df['end_y'] = new_df['y'] + new_df['dy'] * new_df['length']
        


        return new_df
        
        
