"""Fast exact-ground-state calibration for the full-tensor HUBO.

The expensive state features are precomputed once. CMA-ES then operates in the
five-dimensional log10(lambda) space and evaluates a whole population with one
matrix multiplication per generation. No VQE is run inside the training loop;
VQE is reserved for one final quantum-vs-exact validation after learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np

from .higher_moment_extension import SourceMomentData
from .source_hubo_models import SourceHuboConfig, source_hubo_state_table

LAMBDA_NAMES: Tuple[str, ...] = ("return", "variance", "skewness", "kurtosis", "budget")
LAMBDA_BOUNDS = np.asarray([[0.1, 5.0], [0.1, 5.0], [0.001, 0.1], [0.0001, 0.01], [10.0, 1000.0]], dtype=float)
LOG10_BOUNDS = np.log10(LAMBDA_BOUNDS)
SOURCE_WARM_START = np.asarray([1.23736853, 1.21198716, 0.0136952480, 0.00118943202, 161.126224], dtype=float)

@dataclass(frozen=True)
class TrainingConfig:
    generations: int = 30
    population_size: int = 12
    seed: int = 42
    initial_sigma: float = 0.45
    sharpe_weight: float = 1.0
    budget_weight: float = 25.0
    exact_match_bonus: float = 0.05
    patience: int = 10
    def __post_init__(self) -> None:
        if self.generations <= 0 or self.population_size < 4: raise ValueError("generations must be positive and population_size >= 4")
        if self.initial_sigma <= 0.0 or self.patience <= 0: raise ValueError("initial_sigma and patience must be positive")

@dataclass(frozen=True)
class FeatureCache:
    features: np.ndarray
    sharpe: np.ndarray
    budget_breach: np.ndarray
    feasible: np.ndarray
    financial_ground_truth_index: int
    financial_ground_truth_sharpe: float
    state_count: int

def build_feature_cache(data: SourceMomentData) -> FeatureCache:
    cfg = SourceHuboConfig(mode="budget_aligned", lambda_return=1.0, lambda_variance=1.0, lambda_skewness=1.0, lambda_kurtosis=1.0, lambda_budget=1.0)
    table = source_hubo_state_table(data, cfg)
    features = np.column_stack([table["return_energy"], table["variance_energy"], table["skewness_energy"], table["kurtosis_energy"], table["budget_energy"]]).astype(float, copy=False)
    feasible = np.asarray(table["admissible"], dtype=bool)
    feasible_idx = np.flatnonzero(feasible)
    financial_idx = int(feasible_idx[np.argmax(table["sharpe_ratio"][feasible_idx])])
    return FeatureCache(np.ascontiguousarray(features), np.asarray(table["sharpe_ratio"], dtype=float), np.asarray(table["budget_breach"], dtype=float), feasible, financial_idx, float(table["sharpe_ratio"][financial_idx]), int(features.shape[0]))

def _population_losses(log10_population: np.ndarray, cache: FeatureCache, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambdas = np.power(10.0, log10_population)
    energies = cache.features @ lambdas.T
    ground_indices = np.argmin(energies, axis=0)
    selected_sharpe = cache.sharpe[ground_indices]
    selected_breach = cache.budget_breach[ground_indices]
    sharpe_error = np.abs(selected_sharpe - cache.financial_ground_truth_sharpe)
    breach_units = selected_breach / 1250.0
    losses = config.sharpe_weight * sharpe_error + config.budget_weight * breach_units * breach_units - config.exact_match_bonus * (ground_indices == cache.financial_ground_truth_index).astype(float)
    return losses, ground_indices.astype(int), lambdas

def train_exact_ground_lambdas(data: SourceMomentData, config: TrainingConfig | None = None) -> Dict[str, Any]:
    cfg = TrainingConfig() if config is None else config
    started = perf_counter(); cache_started = perf_counter(); cache = build_feature_cache(data); cache_seconds = perf_counter() - cache_started
    n = len(LAMBDA_NAMES); popsize = int(cfg.population_size); mu = popsize // 2
    raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); weights = raw_weights / raw_weights.sum(); mueff = float(1.0 / np.sum(weights**2))
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n); cs = (mueff + 2.0) / (n + mueff + 5.0); c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff)); damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    rng = np.random.default_rng(cfg.seed); mean = np.log10(SOURCE_WARM_START); sigma = float(cfg.initial_sigma); cov = np.eye(n, dtype=float); pc = np.zeros(n); ps = np.zeros(n)
    warm_loss, warm_ground, _ = _population_losses(mean[None, :], cache, cfg); best_loss = float(warm_loss[0]); best_log = mean.copy(); best_ground = int(warm_ground[0]); best_generation = 0; stale = 0
    history = [{"generation": 0, "best_loss": best_loss, "global_best_loss": best_loss, "ground_state_index": best_ground, "ground_feasible": bool(cache.feasible[best_ground]), "ground_sharpe": float(cache.sharpe[best_ground]), "sigma": sigma}]; evaluations = 1
    if best_ground != cache.financial_ground_truth_index:
        for generation in range(cfg.generations):
            eigvals, eigvecs = np.linalg.eigh(cov); eigvals = np.maximum(eigvals, 1e-12); transform = eigvecs @ np.diag(np.sqrt(eigvals)); inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            z = rng.standard_normal((popsize, n)); y = z @ transform.T; candidates = np.clip(mean[None, :] + sigma * y, LOG10_BOUNDS[:, 0], LOG10_BOUNDS[:, 1]); y = (candidates - mean[None, :]) / max(sigma, 1e-12)
            losses, grounds, _ = _population_losses(candidates, cache, cfg); evaluations += popsize; order = np.argsort(losses); elite = order[:mu]; generation_best = int(order[0])
            if float(losses[generation_best]) < best_loss - 1e-12:
                best_loss = float(losses[generation_best]); best_log = candidates[generation_best].copy(); best_ground = int(grounds[generation_best]); best_generation = generation + 1; stale = 0
            else: stale += 1
            old_mean = mean.copy(); mean = np.sum(weights[:, None] * candidates[elite], axis=0); y_w = (mean - old_mean) / max(sigma, 1e-12)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt @ y_w); ps_norm = float(np.linalg.norm(ps)); hsig = float(ps_norm / np.sqrt(max(1e-15, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1)))) < (1.4 + 2.0 / (n + 1.0)) * chi_n)
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
            rank_mu = np.zeros((n, n));
            for weight, step in zip(weights, y[elite]): rank_mu += weight * np.outer(step, step)
            cov = (1.0 - c1 - cmu) * cov + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * cov) + cmu * rank_mu; cov = 0.5 * (cov + cov.T)
            sigma *= float(np.exp((cs / damps) * (ps_norm / chi_n - 1.0))); sigma = float(np.clip(sigma, 0.03, 1.2))
            history.append({"generation": generation + 1, "best_loss": float(losses[generation_best]), "global_best_loss": best_loss, "ground_state_index": int(grounds[generation_best]), "ground_feasible": bool(cache.feasible[grounds[generation_best]]), "ground_sharpe": float(cache.sharpe[grounds[generation_best]]), "sigma": sigma})
            if best_ground == cache.financial_ground_truth_index or stale >= cfg.patience: break
    learned = np.power(10.0, best_log); final_losses, final_grounds, _ = _population_losses(best_log[None, :], cache, cfg); final_ground = int(final_grounds[0])
    return {"status": "completed", "method": "exact-ground-state CMA-ES", "lambdas": {name: float(value) for name, value in zip(LAMBDA_NAMES, learned)}, "lambda_vector": [float(x) for x in learned], "loss": float(final_losses[0]), "evaluations": int(evaluations), "generations_completed": int(max(0, len(history)-1)), "best_generation": int(best_generation), "state_count": cache.state_count, "feature_cache_seconds": float(cache_seconds), "training_seconds": float(perf_counter()-started), "financial_ground_truth_index": int(cache.financial_ground_truth_index), "financial_ground_truth_sharpe": float(cache.financial_ground_truth_sharpe), "learned_ground_state_index": final_ground, "learned_ground_state_sharpe": float(cache.sharpe[final_ground]), "learned_ground_state_budget_breach": float(cache.budget_breach[final_ground]), "learned_ground_state_feasible": bool(cache.feasible[final_ground]), "exact_financial_ground_truth_recovered": bool(final_ground == cache.financial_ground_truth_index), "history": history, "configuration": {"generations": cfg.generations, "population_size": cfg.population_size, "seed": cfg.seed, "initial_sigma": cfg.initial_sigma, "patience": cfg.patience, "warm_start": SOURCE_WARM_START.tolist()}}

def source_hubo_config_from_training(training: Dict[str, Any]) -> SourceHuboConfig:
    lmb = training["lambdas"]
    return SourceHuboConfig(mode="budget_aligned", lambda_return=float(lmb["return"]), lambda_variance=float(lmb["variance"]), lambda_skewness=float(lmb["skewness"]), lambda_kurtosis=float(lmb["kurtosis"]), lambda_budget=float(lmb["budget"]))
