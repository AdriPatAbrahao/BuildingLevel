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

        # Extrai grupos de simetria (opcional)
        if 'group_id' in self.seed_df.columns:
            gids = []
            for i, v in enumerate(self.seed_df['group_id']):
                if pd.isna(v) or (isinstance(v, str) and v.strip() == ''):
                    gids.append(f"__solo_{i}")
                else:
                    gids.append(str(v))
            self.seed_df['group_id'] = gids
        else:
            self.seed_df['group_id'] = [f"__solo_{i}" for i in range(len(self.seed_df))]
        groups = self.seed_df.groupby('group_id').indices

        # Reduz dimensionalidade para drivers por grupo
        self.group_keys = list(groups.keys())
        self.group_indices = [groups[k] for k in self.group_keys]
        self.num_variables = len(self.group_keys)

        # Bounds por grupo: lower = max(length_inicial), upper = min(maxlength)
        initial_lengths = self.seed_df['length'].astype(float).values
        max_lengths = self.seed_df['maxlength'].astype(float).values
        self.lower_bounds = np.array([initial_lengths[idxs].max() for idxs in self.group_indices], dtype=float)
        self.upper_bounds = np.array([max_lengths[idxs].min() for idxs in self.group_indices], dtype=float)
        for i in range(len(self.lower_bounds)):
            if self.upper_bounds[i] < self.lower_bounds[i]:
                self.upper_bounds[i] = self.lower_bounds[i]
        # Chute inicial por grupo: usar lower_bounds (garantidamente viável)
        self.initial_guess = self.lower_bounds.copy()

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
        # Aplica drivers por grupo
        for i, idxs in enumerate(self.group_indices):
            val = float(vector[i])
            new_df.loc[idxs, 'length'] = val

        new_df['end_x'] = new_df['x'] + new_df['dx'] * new_df['length']
        new_df['end_y'] = new_df['y'] + new_df['dy'] * new_df['length']
        


        return new_df
        
        
