# Em run_optimization.py

import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# Importe as novas classes de otimização
from optimization.design_space import DesignSpace
from optimization.objective_function import ObjectiveFunction
from optimization.optimizer import GeneticOptimizer

from inference import BuildingInference
from visualization.nn_diagnostics import run_optimization_diagnostics, OptimizationDiagnosticsPlotter

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

        # --- ETAPA 2b: Avaliar métricas do seed (ponto inicial) ---
        print("\n[PASSO 2b/4] Avaliando métricas do seed para comparação...")
        seed_metrics_raw = objective_function.compute_metrics(design_space.initial_guess)
        seed_metrics = {
            "cost":             seed_metrics_raw.get("cost"),
            "steel":            seed_metrics_raw.get("steel"),
            "concrete":         seed_metrics_raw.get("concrete"),
            "form_area":        seed_metrics_raw.get("form_area"),
            "cost_steel_rs":    seed_metrics_raw.get("cost_steel_rs"),
            "cost_concrete_rs": seed_metrics_raw.get("cost_concrete_rs"),
            "cost_form_rs":     seed_metrics_raw.get("cost_form_rs"),
        }
        print(f"   Seed → custo: R$ {seed_metrics['cost']:,.2f} | "
              f"aço: {seed_metrics['steel']:.1f} kgf | "
              f"concreto: {seed_metrics['concrete']:.3f} m³ | "
              f"forma: {seed_metrics['form_area']:.2f} m²")

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

            # --- Diagnósticos de otimização ---
            optimal_metrics_raw = objective_function.compute_metrics(final_optimal_vector)
            optimal_metrics = {
                "cost":             optimal_metrics_raw.get("cost"),
                "steel":            optimal_metrics_raw.get("steel"),
                "concrete":         optimal_metrics_raw.get("concrete"),
                "form_area":        optimal_metrics_raw.get("form_area"),
                "cost_steel_rs":    optimal_metrics_raw.get("cost_steel_rs"),
                "cost_concrete_rs": optimal_metrics_raw.get("cost_concrete_rs"),
                "cost_form_rs":     optimal_metrics_raw.get("cost_form_rs"),
            }

            # --- Salva o breakdown de custo (R$) seed vs. ótimo ---
            _breakdown_metrics = [
                ("cost",             "Custo total (R$)"),
                ("cost_steel_rs",    "Custo aço (R$)"),
                ("cost_concrete_rs", "Custo concreto (R$)"),
                ("cost_form_rs",     "Custo forma (R$)"),
                ("steel",            "Aço (kgf)"),
                ("concrete",         "Concreto (m³)"),
                ("form_area",        "Forma (m²)"),
            ]

            def _reduction_pct(seed_val, opt_val):
                if not seed_val:
                    return None
                return (seed_val - opt_val) / seed_val * 100

            cost_breakdown = {
                "seed": seed_metrics,
                "optimal": optimal_metrics,
                "reduction_pct": {
                    key: _reduction_pct(seed_metrics.get(key), optimal_metrics.get(key))
                    for key, _ in _breakdown_metrics
                },
            }

            breakdown_json_path = output_path.parent / "cost_breakdown.json"
            breakdown_csv_path = output_path.parent / "cost_breakdown.csv"
            try:
                with open(breakdown_json_path, 'w', encoding='utf-8') as f:
                    json.dump(cost_breakdown, f, ensure_ascii=False, indent=2)
                with open(breakdown_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["metric", "label", "seed", "optimal", "reduction_pct"])
                    for key, label in _breakdown_metrics:
                        writer.writerow([
                            key, label,
                            seed_metrics.get(key), optimal_metrics.get(key),
                            cost_breakdown["reduction_pct"].get(key),
                        ])
                print(f"Breakdown de custo salvo em: '{breakdown_json_path}' e '{breakdown_csv_path}'")
            except Exception as _bd_exc:
                print(f"[CostBreakdown] Falha ao salvar breakdown de custo: {_bd_exc}")

            plots_dir = output_path.parent / "plots"
            log_path  = output_path.parent / "optimization_log.json"
            try:
                run_optimization_diagnostics(
                    output_dir=plots_dir,
                    log_path=log_path,
                    seed_metrics=seed_metrics,
                    optimal_metrics=optimal_metrics,
                )
                print(f"Gráficos de otimização salvos em: '{plots_dir}'")
                print(
                    "\nPara validar o ponto ótimo com o TQS e gerar o gráfico de verificação, "
                    "execute após rodar o TQS:\n"
                    "  from visualization.nn_diagnostics import OptimizationDiagnosticsPlotter\n"
                    f"  p = OptimizationDiagnosticsPlotter('{plots_dir}', '{log_path}')\n"
                    f"  p.plot_surrogate_vs_tqs(surrogate_steel={optimal_metrics['steel']:.1f}, tqs_steel=<VALOR_TQS>)"
                )
            except Exception as _diag_exc:
                print(f"[OptDiagnostics] Skipped: {_diag_exc}")

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

