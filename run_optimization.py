# Em run_optimization.py

import numpy as np
import pandas as pd
from pathlib import Path

# Importe as novas classes de otimização
from optimization.design_space import DesignSpace
from optimization.objective_function import ObjectiveFunction
from optimization.optimizer import GeneticOptimizer

# Importe a classe de inferência que você já tem
from inference import BuildingInference

def main():
    """
    Orquestra o fluxo completo de otimização da edificação.
    """
    print("======================================================")
    print("=      INICIANDO PROCESSO DE OTIMIZAÇÃO DE CUSTO     =")
    print("======================================================")

    try:
        # --- ETAPA 1: Carregar os componentes necessários ---
        print("\n[PASSO 1/4] Carregando modelo substituto e espaço de busca...")
        
        # Carrega o modelo treinado e a pipeline de features
        inference_runner = BuildingInference()
        
        # Carrega a definição do espaço de busca a partir do CSV semente
        # Preferir Building1c.csv, se disponível
        from config import paths
        seed_candidate = paths.DATA_DIR / "Building1c.csv"
        if seed_candidate.exists():
            print(f"   - Usando CSV semente: '{seed_candidate}'")
            design_space = DesignSpace(seed_csv_path=seed_candidate)
        else:
            print("   - 'Building1c.csv' não encontrado. Usando CSV padrão em config.paths.SEED_VECTOR_CSV.")
            design_space = DesignSpace()

        # --- ETAPA 2: Configurar a função objetivo ---
        print("\n[PASSO 2/4] Configurando a função objetivo (custo)...")
        objective_function = ObjectiveFunction(design_space, inference_runner)

        # --- ETAPA 3: Executar a otimização ---
        print("\n[PASSO 3/4] Executando o algoritmo de otimização...")
        optimizer = GeneticOptimizer(objective_function, design_space)
        result = optimizer.run()

        # --- ETAPA 4: Apresentar e salvar os resultados ---
        print("\n[PASSO 4/4] Processando resultados da otimização...")
        if result.success:
            continuous_optimal_vector = result.x
            
            # --- INÍCIO DA MODIFICAÇÃO ---
            # Discretiza o vetor final para obter a solução de projeto real
            final_optimal_vector = objective_function._discretize_vector(continuous_optimal_vector)
            # Recalcula o custo final com o vetor discretizado para garantir consistência
            optimal_cost = objective_function.calculate_cost(final_optimal_vector)
            # --- FIM DA MODIFICAÇÃO ---

            print("\n--- RESULTADO ÓTIMO ENCONTRADO ---")
            print(f"  -> Custo Mínimo Previsto: R$ {optimal_cost:,.2f}")
            print(f"  -> Vetor de Comprimentos Ótimo (cm):")
            print("     " + np.array2string(final_optimal_vector, formatter={'float_kind':lambda x: "%.2f" % x}))

            # Salvar a solução ótima (já discretizada) em um CSV
            output_path = Path("outputs/results/solucao_otima.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            optimal_geometry_df = design_space.create_geometry_from_vector(final_optimal_vector)
            optimal_geometry_df.to_csv(output_path, index=False, sep=';', decimal=',')
            
            print(f"\nSolução ótima salva em: '{output_path}'")
            print("Use este arquivo no seu script 'inference.py' para validar o resultado com o TQS.")

        else:
            print("\n--- ATENÇÃO: A otimização não convergiu com sucesso. ---")
            print(f"Mensagem: {result.message}")

    except Exception as e:
        print(f"\n--- ERRO CRÍTICO NO PROCESSO DE OTIMIZAÇÃO ---")
        import traceback
        print(f"Erro: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    # Adicione esta verificação para garantir que a paralelização funcione corretamente
    # em alguns sistemas operacionais (como Windows).
    main()

