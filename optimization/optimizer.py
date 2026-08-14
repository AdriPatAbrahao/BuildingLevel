# Em optimization/optimizer.py

import time
import numpy as np
from dataclasses import dataclass
from config import paths
from config.settings import ObjectiveConfig
import json
import csv

from optimization.objective_function import ObjectiveFunction
from optimization.design_space import DesignSpace


@dataclass
class OptimizeResult:
    """Result of a genetic optimization run."""
    success: bool
    x: np.ndarray
    fun: float
    nit: int
    message: str = ""


class GeneticOptimizer:
    """
    Simple Genetic Algorithm (GA) minimizing the objective function.

    - Initial population respects `DesignSpace` bounds
    - Tournament selection
    - Blend arithmetic crossover
    - Gaussian mutation clipped to bounds
    - Elitism
    """

    def __init__(self, objective_function: ObjectiveFunction, design_space: DesignSpace,
                 pop_size: int = 40,
                 generations: int = 80,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 tournament_k: int = 3,
                 patience: int = 20,
                 max_time_sec: int | None = None,
                 min_steel_kg: float | None = None,
                 max_invalid_prob: float | None = None,
                 random_state: int | None = None):
        self.objective_func = objective_function
        self.design_space = design_space
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.patience = patience
        self.rng = np.random.default_rng(random_state)
        self.max_time_sec = max_time_sec if max_time_sec is not None else getattr(ObjectiveConfig, 'MAX_TIME_SEC', 600)
        self.min_steel_kg = min_steel_kg if min_steel_kg is not None else getattr(ObjectiveConfig, 'STOP_MIN_STEEL_KG', 0.0)
        self.max_invalid_prob = max_invalid_prob if max_invalid_prob is not None else getattr(ObjectiveConfig, 'STOP_MAX_INVALID_PROB', 0.1)

        # Pré-calcula vetores úteis
        self.lower = np.asarray(self.design_space.lower_bounds, dtype=float)
        self.upper = np.asarray(self.design_space.upper_bounds, dtype=float)
        self.dim = self.lower.size

    def _init_population(self) -> np.ndarray:
        """Initialize a population within bounds including the initial guess."""
        pop = self.rng.uniform(self.lower, self.upper, size=(self.pop_size, self.dim))
        # Garante que o indivíduo inicial esteja presente
        init = np.asarray(self.design_space.initial_guess, dtype=float)
        pop[0] = np.clip(init, self.lower, self.upper)
        return pop

    def _evaluate(self, pop: np.ndarray) -> tuple[np.ndarray, list]:
        """Evaluate the objective for each individual sequentially.

        Returns
        -------
        costs : np.ndarray
            Cost for each individual.
        metrics_cache : list[dict | None]
            Metrics dict cached from the last ``compute_metrics`` call inside
            ``calculate_cost``, indexed by individual position.  Avoids a
            second evaluation of the best individual when logging generation
            statistics.
        """
        costs = np.empty(pop.shape[0], dtype=float)
        metrics_cache: list = [None] * pop.shape[0]
        for i, x in enumerate(pop):
            costs[i] = self.objective_func.calculate_cost(x)
            metrics_cache[i] = self.objective_func._last_metrics
        return costs, metrics_cache

    def _tournament_select(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Pick the best individual among `k` random candidates."""
        idx = self.rng.integers(0, pop.shape[0], size=self.tournament_k)
        best = idx[np.argmin(costs[idx])]
        return pop[best].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Blend crossover producing two children and clipping to bounds."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        alpha = self.rng.uniform(0.0, 1.0, size=self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        # Respeita limites
        return np.clip(child1, self.lower, self.upper), np.clip(child2, self.lower, self.upper)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation with scale proportional to the bounds range."""
        if self.rng.random() > self.mutation_rate:
            return individual
        # Escala de mutação proporcional ao intervalo
        scale = 0.2 * (self.upper - self.lower)
        noise = self.rng.normal(loc=0.0, scale=scale)
        mutated = individual + noise
        return np.clip(mutated, self.lower, self.upper)

    def run(self) -> OptimizeResult:
        """
        Execute the GA loop and return the best individual found.

        Returns
        -------
        OptimizeResult
            Optimization result including best vector, cost and iterations.

        Examples
        --------
        >>> opt = GeneticOptimizer(obj, ds, pop_size=40, generations=80)
        >>> res = opt.run()
        """
        print("\n--- Iniciando Otimização Genética ---")

        pop = self._init_population()
        costs, _ = self._evaluate(pop)

        best_idx = int(np.argmin(costs))
        best_x = pop[best_idx].copy()
        best_cost = float(costs[best_idx])

        start = time.time()
        no_improve = 0
        logs = []
        prev_best_cost = None
        prev_best_metrics = None

        for gen in range(1, self.generations + 1):
            new_pop = []
            # Elitismo: mantém o melhor
            new_pop.append(best_x.copy())

            # Reproduz até preencher a população
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, costs)
                p2 = self._tournament_select(pop, costs)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    c2 = self._mutate(c2)
                    new_pop.append(c2)

            pop = np.vstack(new_pop)
            costs, metrics_cache = self._evaluate(pop)

            # Atualiza melhor
            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            gen_best_x = pop[gen_best_idx].copy()
            # Reuse cached metrics — avoids a second full evaluation of the best
            # individual (was previously calling compute_metrics again here).
            gen_metrics = metrics_cache[gen_best_idx] or self.objective_func.compute_metrics(gen_best_x)

            improved = gen_best_cost + 1e-8 < best_cost
            if improved:
                best_cost = gen_best_cost
                best_x = gen_best_x
                no_improve = 0
            else:
                no_improve += 1

            header = f"\n=== Iteração {gen} ==="
            print(header)
            print(f"Custo Atual: R$ {gen_best_cost:,.2f}")
            if prev_best_cost is not None:
                delta = gen_best_cost - prev_best_cost
                flag = "▲" if delta < 0 else ("=" if delta == 0 else "▼")
                print(f"Comparação vs anterior: {flag} ΔR$ {abs(delta):,.2f}")
            print("Steel (kg):", f"{gen_metrics['steel']:.2f}")
            print("Forma (m²):", f"{gen_metrics.get('form_area', 0.0):.2f}")
            print("Validade (prob_invalid):", f"{gen_metrics['prob_invalid'] if gen_metrics['prob_invalid'] is not None else 'N/A'}")
            print(f"Sem melhora consecutiva: {no_improve}")

            logs.append({
                "iteration": gen,
                "cost": gen_best_cost,
                "steel": gen_metrics.get("steel"),
                "concrete": gen_metrics.get("concrete"),
                "form_area": gen_metrics.get("form_area"),
                "prob_invalid": gen_metrics.get("prob_invalid"),
                "improved": improved,
                "no_improve": no_improve,
                "elapsed_sec": time.time() - start
            })
            prev_best_cost = gen_best_cost
            prev_best_metrics = gen_metrics

            if no_improve >= self.patience:
                print("Critério de parada: estagnação.")
                break
            if time.time() - start >= self.max_time_sec:
                print("Critério de parada: tempo máximo.")
                break
            if gen_metrics.get("steel") is not None and gen_metrics.get("prob_invalid") is not None:
                if gen_metrics["steel"] <= self.min_steel_kg and gen_metrics["prob_invalid"] <= self.max_invalid_prob:
                    print("Critério de parada: metas mínimas atingidas.")
                    break

        elapsed = time.time() - start
        print(f"--- Otimização Concluída em {elapsed:.2f} segundos ---")
        json_path = paths.RESULTS_DIR / "optimization_log.json"
        csv_path = paths.RESULTS_DIR / "optimization_log.csv"
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
                w.writeheader()
                w.writerows(logs)
            print(f"Logs exportados: {json_path} | {csv_path}")
        except Exception:
            pass

        return OptimizeResult(success=True, x=best_x, fun=best_cost, nit=gen,
                               message="Convergência por critérios configurados")

