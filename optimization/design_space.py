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
            raw = self.seed_df['group_id'].to_numpy(dtype=object)
            gids = [
                f"__solo_{i}" if (pd.isna(v) or (isinstance(v, str) and v.strip() == ''))
                else str(v)
                for i, v in enumerate(raw)
            ]
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

        # Pre-cache numpy arrays for fast vectorized geometry reconstruction.
        # Avoids full DataFrame.copy() in the hot path (called thousands of times
        # during optimization).
        self._x          = self.seed_df['x'].to_numpy(dtype=float)
        self._y          = self.seed_df['y'].to_numpy(dtype=float)
        self._dx         = self.seed_df['dx'].to_numpy(dtype=float)
        self._dy         = self.seed_df['dy'].to_numpy(dtype=float)
        self._base_lengths  = self.seed_df['length'].to_numpy(dtype=float)
        self._maxlengths    = self.seed_df['maxlength'].to_numpy(dtype=float)
        self._group_id_vals = self.seed_df['group_id'].to_numpy(dtype=object)
        # Static DataFrame: columns that never change (no length/end_x/end_y).
        _drop = [c for c in ('length', 'end_x', 'end_y') if c in self.seed_df.columns]
        self._static_df = self.seed_df.drop(columns=_drop) if _drop else self.seed_df

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

        # Build lengths array with numpy (avoids pandas .loc indexing per group).
        lengths = self._base_lengths.copy()
        for i, idxs in enumerate(self.group_indices):
            lengths[idxs] = float(vector[i])

        # Copy only static columns; assign computed columns as numpy arrays.
        new_df = self._static_df.copy()
        new_df['length'] = lengths
        new_df['end_x']  = self._x + self._dx * lengths
        new_df['end_y']  = self._y + self._dy * lengths

        return new_df

    def segments_from_vector(self, vector: np.ndarray) -> list:
        """
        Create the list of segment dicts directly from a length vector — no
        DataFrame allocation and no CSV serialisation/parsing.

        This is the fast path used during optimisation.  Equivalent to calling
        ``create_geometry_from_vector`` + ``LengthProcessor.read_length_from_csv``
        but avoids all I/O overhead.

        Returns
        -------
        list[dict]
            Each dict has ``start``, ``end``, ``length``, ``maxlength``,
            ``binary`` and ``group_id`` keys, matching the format produced by
            ``LengthProcessor.read_length_from_csv``.
        """
        if len(vector) != self.num_variables:
            raise ValueError(f"Vetor de entrada tem {len(vector)} elementos, mas o esperado era {self.num_variables}.")

        lengths = self._base_lengths.copy()
        for i, idxs in enumerate(self.group_indices):
            lengths[idxs] = float(vector[i])

        end_x = self._x + self._dx * lengths
        end_y = self._y + self._dy * lengths

        segments = []
        for i in range(len(self._x)):
            gid = self._group_id_vals[i]
            segments.append({
                'start':     (float(self._x[i]),   float(self._y[i])),
                'end':       (float(end_x[i]),      float(end_y[i])),
                'length':    float(lengths[i]),
                'maxlength': float(self._maxlengths[i]),
                'binary':    1,
                'group_id':  str(gid),
            })
        return segments

