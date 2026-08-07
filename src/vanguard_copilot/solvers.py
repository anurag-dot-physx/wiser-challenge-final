"""Efficient and audited classical solvers for the quadratic portfolio model."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from .model import AssetClassData, InvestorProfile, PortfolioMetrics, feasible_weight_mask, objective_terms, portfolio_metrics, quadratic_objectives


@dataclass(frozen=True)
class SolverResult:
    method: str
    weights: np.ndarray
    metrics: PortfolioMetrics
    runtime_seconds: float
    status: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ReducedProblem:
    data: AssetClassData
    original_indices: Tuple[int, ...]
    profile: InvestorProfile
    step: float
    bits_per_asset: int

    @property
    def target_units(self) -> int:
        return grid_units(self.step)

    @property
    def encoding_max_units(self) -> int:
        return 2**self.bits_per_asset - 1


def grid_units(step: float) -> int:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    reciprocal = 1.0 / step
    units = int(round(reciprocal))
    if units <= 0 or not np.isclose(reciprocal, units, atol=1e-12, rtol=1e-12):
        raise ValueError("step must divide one exactly within numerical tolerance.")
    return units


def linear_constraint_system(data: AssetClassData, profile: InvestorProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    lower = []
    upper = []
    rows.append(np.ones(data.n_assets)); lower.append(1.0); upper.append(1.0)
    row = np.zeros(data.n_assets); row[list(data.groups["equity"])] = 1.0
    rows.append(row); lower.append(-np.inf); upper.append(profile.equity_max)
    row = np.zeros(data.n_assets); row[list(data.groups["defensive"])] = 1.0
    rows.append(row); lower.append(profile.defensive_min); upper.append(np.inf)
    row = np.zeros(data.n_assets); row[list(data.groups["alternatives"])] = 1.0
    rows.append(row); lower.append(-np.inf); upper.append(profile.alternatives_max)
    return np.asarray(rows, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _find_feasible_point(data: AssetClassData, profile: InvestorProfile) -> np.ndarray:
    A, lower, upper = linear_constraint_system(data, profile)
    equality = np.isclose(lower, upper, atol=1e-14) & np.isfinite(lower) & np.isfinite(upper)
    A_eq = A[equality] if np.any(equality) else None
    b_eq = lower[equality] if np.any(equality) else None
    inequality_rows = []
    inequality_rhs = []
    for row, lo, hi in zip(A[~equality], lower[~equality], upper[~equality]):
        if np.isfinite(hi): inequality_rows.append(row); inequality_rhs.append(hi)
        if np.isfinite(lo): inequality_rows.append(-row); inequality_rhs.append(-lo)
    A_ub = np.asarray(inequality_rows, dtype=float) if inequality_rows else None
    b_ub = np.asarray(inequality_rhs, dtype=float) if inequality_rhs else None
    result = linprog(np.zeros(data.n_assets), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, profile.asset_max)] * data.n_assets, method="highs")
    if not result.success:
        raise ValueError(f"Portfolio guardrails are infeasible: {result.message}")
    return np.asarray(result.x, dtype=float)


def _constraint_violation(weights: np.ndarray, A: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    values = A @ weights
    lower_violation = np.max(np.where(np.isfinite(lower), np.maximum(lower - values, 0.0), 0.0))
    upper_violation = np.max(np.where(np.isfinite(upper), np.maximum(values - upper, 0.0), 0.0))
    return float(max(lower_violation, upper_violation))


def solve_continuous(data: AssetClassData, profile: InvestorProfile) -> SolverResult:
    started = perf_counter()
    H, g, c = objective_terms(data, profile)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(H)))
    if min_eigenvalue < -1e-10:
        raise ValueError("The quadratic objective is not convex.")
    A, lower, upper = linear_constraint_system(data, profile)
    x0 = _find_feasible_point(data, profile)
    def objective(weights: np.ndarray) -> float: return float(weights @ H @ weights + g @ weights + c)
    def gradient(weights: np.ndarray) -> np.ndarray: return 2.0 * H @ weights + g
    equality = LinearConstraint(A[:1], lower[:1], upper[:1])
    inequalities = LinearConstraint(A[1:], lower[1:], upper[1:])
    bounds = Bounds(np.zeros(data.n_assets), np.full(data.n_assets, profile.asset_max))
    result = minimize(objective, x0, method="SLSQP", jac=gradient, bounds=bounds, constraints=[equality, inequalities], options={"maxiter": 1000, "ftol": 1e-12, "disp": False})
    weights = np.asarray(result.x, dtype=float); weights[np.abs(weights) < 1e-13] = 0.0
    metrics = portfolio_metrics(weights, data, profile)
    violation = _constraint_violation(weights, A, lower, upper)
    status = "optimal" if result.success and metrics.feasible and violation <= 1e-7 else "failed"
    return SolverResult("continuous_convex_qp_slsqp", weights, metrics, perf_counter() - started, status, {
        "message": str(result.message), "iterations": int(result.nit), "objective_evaluations": int(result.nfev),
        "gradient_evaluations": int(getattr(result, "njev", 0)), "minimum_hessian_eigenvalue": min_eigenvalue,
        "maximum_linear_constraint_violation": violation, "convex_problem": True,
    })


def integer_compositions(total: int, parts: int) -> Iterator[Tuple[int, ...]]:
    if total < 0 or parts <= 0: return
    for bars in combinations(range(total + parts - 1), parts - 1):
        previous = -1; values = []
        for bar in (*bars, total + parts - 1):
            values.append(bar - previous - 1); previous = bar
        yield tuple(values)


def _composition_batches(total: int, parts: int, batch_size: int) -> Iterator[np.ndarray]:
    buffer: list[tuple[int, ...]] = []
    for composition in integer_compositions(total, parts):
        buffer.append(composition)
        if len(buffer) >= batch_size:
            yield np.asarray(buffer, dtype=np.int16); buffer.clear()
    if buffer: yield np.asarray(buffer, dtype=np.int16)


def solve_discrete_exact(data: AssetClassData, profile: InvestorProfile, *, step: float = 0.10, batch_size: int = 50_000, max_states: int = 2_000_000, effective_asset_max: float | None = None) -> SolverResult:
    started = perf_counter()
    if batch_size <= 0 or max_states <= 0: raise ValueError("batch_size and max_states must be positive.")
    units = grid_units(step)
    state_count = comb(units + data.n_assets - 1, data.n_assets - 1)
    if state_count > max_states:
        raise ValueError(f"Exact grid contains {state_count:,} states, exceeding max_states={max_states:,}. Use a coarser grid or a mixed-integer solver.")
    evaluation_profile = profile
    if effective_asset_max is not None:
        if effective_asset_max <= 0.0: raise ValueError("effective_asset_max must be positive.")
        evaluation_profile = profile.with_overrides(asset_max=min(profile.asset_max, effective_asset_max))
    H, g, c = objective_terms(data, profile)
    best_weights = None; best_value = float("inf"); feasible_count = 0; enumerated = 0
    for unit_batch in _composition_batches(units, data.n_assets, batch_size):
        enumerated += unit_batch.shape[0]
        weights = unit_batch.astype(float) * step
        mask = feasible_weight_mask(weights, data, evaluation_profile, tolerance=1e-12)
        feasible_count += int(mask.sum())
        if not np.any(mask): continue
        candidates = weights[mask]
        values = quadratic_objectives(candidates, H, g, c)
        local = int(np.argmin(values))
        if float(values[local]) < best_value - 1e-14:
            best_value = float(values[local]); best_weights = candidates[local].copy()
    if best_weights is None: raise RuntimeError("No grid portfolio satisfies the hard constraints.")
    metrics = portfolio_metrics(best_weights, data, profile)
    if not metrics.feasible: raise RuntimeError("Internal error: exact grid solver returned an infeasible portfolio.")
    return SolverResult("discrete_exact_grid_vectorized", best_weights, metrics, perf_counter() - started, "optimal", {
        "step": step, "states_enumerated": enumerated, "feasible_states": feasible_count, "batch_size": batch_size,
        "optimality": "global optimum on the stated grid",
    })


def baseline_result(data: AssetClassData, profile: InvestorProfile) -> SolverResult:
    started = perf_counter(); metrics = portfolio_metrics(data.current_weights, data, profile)
    return SolverResult("current_portfolio_baseline", data.current_weights.copy(), metrics, perf_counter() - started, "feasible" if metrics.feasible else "infeasible_baseline", {})


def make_reduced_problem(data: AssetClassData, profile: InvestorProfile, *, asset_indices: Sequence[int] = (0, 1, 3, 5, 7), step: float = 0.125, bits_per_asset: int = 3) -> ReducedProblem:
    indices = tuple(int(index) for index in asset_indices)
    if len(indices) < 3 or len(set(indices)) != len(indices): raise ValueError("The reduced quantum instance requires at least three distinct assets.")
    if any(index < 0 or index >= data.n_assets for index in indices): raise ValueError("asset_indices contains an out-of-range asset index.")
    if bits_per_asset <= 0: raise ValueError("bits_per_asset must be positive.")
    target_units = grid_units(step)
    if target_units > len(indices) * (2**bits_per_asset - 1): raise ValueError("The binary encoding cannot represent a fully invested portfolio.")
    names = tuple(data.names[index] for index in indices)
    current = data.current_weights[list(indices)].astype(float); current = current / current.sum()
    groups: Dict[str, Tuple[int, ...]] = {name: tuple(reduced_index for reduced_index, original_index in enumerate(indices) if original_index in data.groups[name]) for name in ("equity", "defensive", "alternatives")}
    missing = [name for name, members in groups.items() if not members]
    if missing: raise ValueError("The reduced asset set must represent every guardrail group; missing groups: " + ", ".join(missing))
    reduced_data = AssetClassData(
        names=names, expected_returns=data.expected_returns[list(indices)], covariance=data.covariance[np.ix_(indices, indices)],
        income_yields=data.income_yields[list(indices)], transaction_costs=data.transaction_costs[list(indices)], current_weights=current,
        scenario_losses=data.scenario_losses[:, list(indices)], scenario_names=data.scenario_names, groups=groups,
    )
    return ReducedProblem(reduced_data, indices, profile, step, bits_per_asset)


def solve_reduced_exact(problem: ReducedProblem) -> SolverResult:
    encoding_max = problem.encoding_max_units * problem.step
    result = solve_discrete_exact(problem.data, problem.profile, step=problem.step, max_states=1_000_000, effective_asset_max=encoding_max)
    return SolverResult("reduced_exact_grid", result.weights, result.metrics, result.runtime_seconds, result.status, {
        **dict(result.metadata), "bits_per_asset": problem.bits_per_asset, "encoding_max_weight": encoding_max,
    })
