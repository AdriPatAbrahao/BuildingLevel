# Em optimization/optimizer.py

import time
import numpy as np
from dataclasses import dataclass

from optimization.objective_function import ObjectiveFunction
from optimization.design_space import DesignSpace


@dataclass
class OptimizeResult:
    success: bool
    x: np.ndarray
    fun: float
    nit: int
    message: str = ""


class GeneticOptimizer:
    """
    Implementa um Algoritmo Genético simples (sem SciPy) para minimizar a função objetivo.
    - População inicial respeita limites do DesignSpace
    - Seleção por torneio
    - Crossover aritmético (blend)
    - Mutação gaussiana com recorte aos limites
    - Elitismo
    """

    def __init__(self, objective_function: ObjectiveFunction, design_space: DesignSpace,
                 pop_size: int = 40,
                 generations: int = 80,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 tournament_k: int = 3,
                 patience: int = 20,
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

        # Pré-calcula vetores úteis
        self.lower = np.asarray(self.design_space.lower_bounds, dtype=float)
        self.upper = np.asarray(self.design_space.upper_bounds, dtype=float)
        self.dim = self.lower.size

    def _init_population(self) -> np.ndarray:
        pop = self.rng.uniform(self.lower, self.upper, size=(self.pop_size, self.dim))
        # Garante que o indivíduo inicial esteja presente
        init = np.asarray(self.design_space.initial_guess, dtype=float)
        pop[0] = np.clip(init, self.lower, self.upper)
        return pop

    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        costs = np.empty(pop.shape[0], dtype=float)
        for i, x in enumerate(pop):
            costs[i] = self.objective_func.calculate_cost(x)
        return costs

    def _tournament_select(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, pop.shape[0], size=self.tournament_k)
        best = idx[np.argmin(costs[idx])]
        return pop[best].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        alpha = self.rng.uniform(0.0, 1.0, size=self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        # Respeita limites
        return np.clip(child1, self.lower, self.upper), np.clip(child2, self.lower, self.upper)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.mutation_rate:
            return individual
        # Escala de mutação proporcional ao intervalo
        scale = 0.2 * (self.upper - self.lower)
        noise = self.rng.normal(loc=0.0, scale=scale)
        mutated = individual + noise
        return np.clip(mutated, self.lower, self.upper)

    def run(self) -> OptimizeResult:
        print("\n--- Iniciando Otimização Genética ---")

        pop = self._init_population()
        costs = self._evaluate(pop)

        best_idx = int(np.argmin(costs))
        best_x = pop[best_idx].copy()
        best_cost = float(costs[best_idx])

        start = time.time()
        no_improve = 0

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
            costs = self._evaluate(pop)

            # Atualiza melhor
            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            gen_best_x = pop[gen_best_idx].copy()

            if gen_best_cost + 1e-8 < best_cost:
                best_cost = gen_best_cost
                best_x = gen_best_x
                no_improve = 0
            else:
                no_improve += 1

            print(f"Geração {gen:3d} | Melhor Custo: R$ {best_cost:,.2f} | Sem melhora: {no_improve}")

            if no_improve >= self.patience:
                print("Critério de parada por estagnação atingido.")
                break

        elapsed = time.time() - start
        print(f"--- Otimização Concluída em {elapsed:.2f} segundos ---")

        return OptimizeResult(success=True, x=best_x, fun=best_cost, nit=gen,
                               message="Convergência por limite de gerações/estagnação")

