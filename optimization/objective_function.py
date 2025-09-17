# Em optimization/objective_function.py

import numpy as np
import io
import traceback

from utils.geometric_calculator import get_geometric_concrete_volume

from inference import BuildingInference
from optimization.design_space import DesignSpace

class ObjectiveFunction:
    def __init__(self, design_space: DesignSpace, inference_runner: BuildingInference):
        # --- PREÇOS E PARÂMETROS ---
        self.PRECO_CONCRETO_M3 = 10.0
        self.PRECO_ACO_KGF = 100.0
        self.COMPRIMENTO_PASSO = 20.0  # PASSO DISCRETO (cm)
        # ---------------------------

        self.design_space = design_space
        self.inference_runner = inference_runner
        
        print("--- Função Objetivo pronta ---")
        print(f"   - Preço Concreto: R$ {self.PRECO_CONCRETO_M3:.2f}/m³")
        print(f"   - Preço Aço:      R$ {self.PRECO_ACO_KGF:.2f}/kg")
        print(f"   - Passo Discreto de Comprimento: {self.COMPRIMENTO_PASSO} cm")

    def _discretize_vector(self, continuous_vector: np.ndarray) -> np.ndarray:
        """
        Arredonda cada valor no vetor para o múltiplo mais próximo do passo definido.
        Ex: Se passo=20, 57.5 -> 60.0; 48.1 -> 40.0; 31.9 -> 40.0
        """
        # (valor / passo) -> arredonda -> * passo
        discretized = np.round(continuous_vector / self.COMPRIMENTO_PASSO) * self.COMPRIMENTO_PASSO
        # clip nos limites do design space
        discretized = np.clip(discretized, self.design_space.lower_bounds, self.design_space.upper_bounds)
        return discretized

    def calculate_cost(self, vector: np.ndarray) -> float:
        """
        Calcula o custo de uma única solução (vetor de comprimentos).
        """
        try:
            # --- INÍCIO DA MODIFICAÇÃO ---
            # >> PASSO 2: Discretize o vetor recebido do otimizador <<
            discretized_vector = self._discretize_vector(vector)
            # --- FIM DA MODIFICAÇÃO ---

            # 1. Converter o vetor JÁ DISCRETIZADO em um DataFrame de geometria
            geometry_df = self.design_space.create_geometry_from_vector(discretized_vector)

            # 2. Simular um arquivo CSV em memória a partir do DataFrame
            csv_buffer = io.StringIO()
            # Usa decimal=',' para casar com o LengthProcessor (reader)
            geometry_df.to_csv(csv_buffer, index=False, sep=';', decimal=',')
            csv_buffer.seek(0)

            # 3. Predição do aço e cálculo geométrico do concreto
            # Reutiliza a função centralizada de inferência, que já:
            #  - Lê o CSV/buffer
            #  - Gera a geometria e extrai features
            #  - Calcula o concreto geométrico
            aco_predito, concreto_predito = self.inference_runner.predict_from_csv(csv_buffer)

            # 4. Calcular o custo total
            custo_total = (aco_predito * self.PRECO_ACO_KGF) + (concreto_predito * self.PRECO_CONCRETO_M3)

            # Penalidade para valores negativos
            if aco_predito < 0 or concreto_predito < 0:
                custo_total += 1_000_000

            return custo_total

        except Exception as e:
            # ... (seu bloco de erro para debug permanece o mesmo) ...
            print("\n--- ERRO DENTRO DA FUNÇÃO OBJETIVO ---")
            print(f"Erro ao avaliar o vetor (contínuo): {vector}")
            print(f"Vetor (discretizado): {discretized_vector if 'discretized_vector' in locals() else 'N/A'}")
            print(f"Tipo de Erro: {type(e).__name__}")
            print(f"Mensagem: {e}")
            print("Traceback completo:")
            print(traceback.format_exc())
            print("---------------------------------------\n")
            return float('inf')
