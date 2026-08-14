# Em optimization/objective_function.py

import numpy as np
import io
import traceback

from utils.geometric_calculator import get_geometric_concrete_volume

from inference import BuildingInference
from optimization.design_space import DesignSpace
from config.settings import ObjectiveConfig

class ObjectiveFunction:
    """
    Cost function combining surrogate steel prediction and geometric concrete volume.

    Parameters
    ----------
    design_space : DesignSpace
        Design space providing bounds and geometry reconstruction utilities.
    inference_runner : BuildingInference
        Inference orchestrator used to predict steel from CSV/buffer.

    Notes
    -----
    Prices and thresholds are read from `ObjectiveConfig` with safe defaults.
    Input vectors are discretized to the nearest multiple of `COMPRIMENTO_PASSO`.
    """
    def __init__(self, design_space: DesignSpace, inference_runner: BuildingInference):
        # --- PREÇOS E PARÂMETROS ---
        # Prefer values from ObjectiveConfig; fallback to previous defaults for backward compatibility
        self.PRECO_CONCRETO_M3 = getattr(ObjectiveConfig, "CONCRETE_PRICE_M3", 10.0)
        self.PRECO_ACO_KGF = getattr(ObjectiveConfig, "STEEL_PRICE_KG", 100.0)
        self.PRECO_FORMA_M2 = getattr(ObjectiveConfig, "FORM_PRICE_M2", 10.0)
        self.COMPRIMENTO_PASSO = getattr(ObjectiveConfig, "LENGTH_STEP_CM", 20.0)  # PASSO DISCRETO (cm)
        self.INVALID_PROB_THRESHOLD = getattr(ObjectiveConfig, "INVALID_PROB_THRESHOLD", 0.5)
        self.INVALID_COST_PENALTY = getattr(ObjectiveConfig, "INVALID_COST_PENALTY", 1_000_000)
        # ---------------------------

        self.design_space = design_space
        self.inference_runner = inference_runner
        # Cache for the last compute_metrics call so that calculate_cost callers
        # can retrieve metrics without a second evaluation of the same vector.
        self._last_metrics: dict | None = None

        print("--- Função Objetivo pronta ---")
        print(f"   - Preço Concreto: R$ {self.PRECO_CONCRETO_M3:.2f}/m³")
        print(f"   - Preço Aço:      R$ {self.PRECO_ACO_KGF:.2f}/kg")
        print(f"   - Preço Forma:    R$ {self.PRECO_FORMA_M2:.2f}/m²")
        print(f"   - Passo Discreto de Comprimento: {self.COMPRIMENTO_PASSO} cm")

    def _discretize_vector(self, continuous_vector: np.ndarray) -> np.ndarray:
        """
        Round each value to the nearest multiple of the configured step.

        Parameters
        ----------
        continuous_vector : np.ndarray
            Continuous vector proposed by the optimizer.

        Returns
        -------
        np.ndarray
            Discretized and clipped vector within design bounds.
        """
        # (valor / passo) -> arredonda -> * passo
        discretized = np.round(continuous_vector / self.COMPRIMENTO_PASSO) * self.COMPRIMENTO_PASSO
        # clip nos limites do design space
        discretized = np.clip(discretized, self.design_space.lower_bounds, self.design_space.upper_bounds)
        return discretized

    def calculate_cost(self, vector: np.ndarray) -> float:
        """
        Compute the total cost for a single candidate vector.

        Parameters
        ----------
        vector : np.ndarray
            Continuous vector of lengths proposed by the optimizer.

        Returns
        -------
        float
            Total cost combining steel (surrogate) and concrete (geometry),
            including penalties for invalid probability and negative values.

        Raises
        ------
        RuntimeError
            Propagated in severe mismatches; otherwise returns `inf` on errors.

        Examples
        --------
        >>> obj = ObjectiveFunction(ds, inference)
        >>> x = np.array([120.0, 60.0, 80.0])
        >>> cost = obj.calculate_cost(x)
        """
        try:
            self._last_metrics = self.compute_metrics(vector)
            return float(self._last_metrics["cost"])
        except Exception as e:
            print("\n--- ERRO DENTRO DA FUNÇÃO OBJETIVO ---")
            print(f"Erro ao avaliar o vetor (contínuo): {vector}")
            print(f"Tipo de Erro: {type(e).__name__}")
            print(f"Mensagem: {e}")
            print(traceback.format_exc())
            self._last_metrics = None
            return float('inf')

    def compute_metrics(self, vector: np.ndarray) -> dict:
        try:
            discretized_vector = self._discretize_vector(vector)

            # Fast path: bypass DataFrame creation and CSV round-trip entirely.
            # Falls back to CSV path if DesignSpace or inference don't support the
            # direct segment interface (e.g. when called from external code).
            if hasattr(self.design_space, 'segments_from_vector') and \
               hasattr(self.inference_runner, 'predict_from_segments'):
                segments = self.design_space.segments_from_vector(discretized_vector)
                steel, concrete, form_area, prob_invalid = self.inference_runner.predict_from_segments(segments)
            else:
                geometry_df = self.design_space.create_geometry_from_vector(discretized_vector)
                csv_buffer = io.StringIO()
                geometry_df.to_csv(csv_buffer, index=False, sep=';', decimal=',')
                csv_buffer.seek(0)
                steel, concrete, form_area, prob_invalid = self.inference_runner.predict_from_csv(csv_buffer)

            cost_steel_rs = steel * self.PRECO_ACO_KGF
            cost_concrete_rs = concrete * self.PRECO_CONCRETO_M3
            cost_form_rs = form_area * self.PRECO_FORMA_M2
            cost = cost_steel_rs + cost_concrete_rs + cost_form_rs
            _raw_thr = getattr(self.inference_runner, 'invalid_threshold', None)
            thr = _raw_thr if _raw_thr is not None else self.INVALID_PROB_THRESHOLD
            if prob_invalid is not None and prob_invalid >= thr:
                cost += self.INVALID_COST_PENALTY
            if steel < 0 or concrete < 0 or form_area < 0:
                cost += 1_000_000
            return {
                "vector": discretized_vector.tolist(),
                "steel": float(steel),
                "concrete": float(concrete),
                "form_area": float(form_area),
                "cost_steel_rs": float(cost_steel_rs),
                "cost_concrete_rs": float(cost_concrete_rs),
                "cost_form_rs": float(cost_form_rs),
                "prob_invalid": float(prob_invalid) if prob_invalid is not None else None,
                "cost": float(cost)
            }
        except Exception as e:
            raise e
