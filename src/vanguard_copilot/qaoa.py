"""Efficient NumPy statevector QAOA for the audited reduced QUBO."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .model import PortfolioMetrics, objective_terms, portfolio_metrics, quadratic_objectives
from .qubo import QuboModel, all_state_energies, audit_qubo, constraint_satisfied_mask, true_feasible_mask
from .solvers import solve_reduced_exact


@dataclass(frozen=True)
class QaoaResult:
    method: str
    selected_weights: np.ndarray
    metrics: PortfolioMetrics
    runtime_seconds: float
    status: str
    angles: np.ndarray
    metadata: Mapping[str, object]


def _apply_rx_layer(state: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    output = state.copy()
    for qubit in range(n_qubits):
        stride = 1 << qubit
        view = output.reshape(-1, 2, stride)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = c * left + s * right
        view[:, 1, :] = s * left + c * right
    return output


def qaoa_state(energies: np.ndarray, angles: Sequence[float], p: int, n_qubits: int) -> np.ndarray:
    angle_array = np.asarray(angles, dtype=float)
    if p <= 0 or angle_array.shape != (2 * p,):
        raise ValueError(f"Expected {2*p} QAOA angles for p={p}.")
    gammas = angle_array[:p]
    betas = angle_array[p:]
    state = np.ones(energies.size, dtype=complex) / np.sqrt(energies.size)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * energies)
        state = _apply_rx_layer(state, beta, n_qubits)
    return state


def run_qaoa(model: QuboModel, *, p: int = 1, maxiter: int = 60, shots: int = 4096, seed: int = 42, restarts: int = 2, model_audit: Mapping[str, object] | None = None) -> QaoaResult:
    if p <= 0 or maxiter <= 0 or shots <= 0 or restarts <= 0:
        raise ValueError("p, maxiter, shots, and restarts must be positive.")
    started = perf_counter()
    initialization_rng = np.random.default_rng(seed)
    shot_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51A7]))
    bits, energies = all_state_energies(model)
    feasible_mask = true_feasible_mask(model, bits)
    constraint_consistent_mask = constraint_satisfied_mask(model, bits)
    if not np.any(feasible_mask):
        raise RuntimeError("The reduced encoding contains no hard-feasible state.")
    scale = max(float(np.std(energies)), 1e-12)
    normalized_energies = (energies - float(np.mean(energies))) / scale

    def expectation(theta: np.ndarray) -> float:
        state = qaoa_state(normalized_energies, theta, p, model.n_qubits)
        probabilities = np.abs(state) ** 2
        return float(probabilities @ normalized_energies)

    best = None
    for restart in range(restarts):
        if restart == 0:
            initial = np.concatenate([np.full(p, 0.35), np.full(p, 0.60)])
        else:
            initial = np.concatenate([
                initialization_rng.uniform(0.0, np.pi, size=p),
                initialization_rng.uniform(0.0, np.pi / 2.0, size=p),
            ])
        result = minimize(expectation, initial, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.25, "tol": 1e-6})
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    state = qaoa_state(normalized_energies, best.x, p, model.n_qubits)
    probabilities = np.abs(state) ** 2
    probabilities /= probabilities.sum()
    counts = shot_rng.multinomial(shots, probabilities)
    observed = np.flatnonzero(counts > 0)
    feasible_observed = observed[feasible_mask[observed]]
    all_weights = bits @ model.allocation_map.T
    reduced_exact = solve_reduced_exact(model.problem)
    exact_mask = feasible_mask & np.all(np.isclose(all_weights, reduced_exact.weights[None, :], atol=1e-12), axis=1)
    if not np.any(exact_mask):
        raise RuntimeError("The exact reduced portfolio is not representable by the QUBO encoding.")
    fallback = feasible_observed.size == 0
    if fallback:
        selected_index = -1
        selected_weights = reduced_exact.weights.copy()
        selected_count = 0
        status = "completed_with_classical_fallback"
    else:
        H, g, c = objective_terms(model.problem.data, model.problem.profile)
        true_values = quadratic_objectives(all_weights[feasible_observed], H, g, c)
        selected_index = int(feasible_observed[int(np.argmin(true_values))])
        selected_weights = all_weights[selected_index]
        selected_count = int(counts[selected_index])
        status = "completed"
    metrics = portfolio_metrics(selected_weights, model.problem.data, model.problem.profile)
    if not metrics.feasible:
        raise RuntimeError("Internal error: reported QAOA allocation violates hard constraints.")
    audit = dict(model_audit) if model_audit is not None else audit_qubo(model)
    exact_probability = float(probabilities[exact_mask].sum())
    ground_mask = np.isclose(energies, np.min(energies), atol=1e-10, rtol=1e-10)
    ground_probability = float(probabilities[ground_mask].sum())
    exact_recovered = bool(np.allclose(selected_weights, reduced_exact.weights, atol=1e-12))
    return QaoaResult(
        method="statevector_qaoa_finite_shot_exact_constraint_qubo",
        selected_weights=np.asarray(selected_weights, dtype=float),
        metrics=metrics,
        runtime_seconds=perf_counter() - started,
        status=status,
        angles=np.asarray(best.x, dtype=float),
        metadata={
            "p": p, "maxiter": maxiter, "restarts": restarts, "shots": shots,
            "optimizer_success": bool(best.success), "optimizer_status": str(best.message),
            "expectation_normalized": float(best.fun),
            "feasible_probability_mass": float(probabilities[feasible_mask].sum()),
            "constraint_consistent_probability_mass": float(probabilities[constraint_consistent_mask].sum()),
            "observed_unique_states": int(observed.size),
            "observed_feasible_states": int(feasible_observed.size),
            "selected_state_index": selected_index,
            "selected_observed_count": selected_count,
            "exact_reduced_probability": exact_probability,
            "qubo_ground_probability": ground_probability,
            "exact_reduced_recovered": exact_recovered,
            "classical_fallback_used": fallback,
            "hard_breaches": 0,
            "qubo_ground_state_hard_feasible": audit["ground_state_hard_feasible"],
            "qubo_ground_matches_exact_reduced": audit["ground_portfolio_matches_reduced_exact"],
            "qubo_ground_constraint_consistent": audit["ground_state_constraint_consistent"],
            "constraint_penalty": model.constraint_penalty,
        },
    )
