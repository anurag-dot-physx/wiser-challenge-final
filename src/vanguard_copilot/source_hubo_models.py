"""Original and budget-aligned full-tensor HUBO extensions."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.optimize import minimize

from .higher_moment_extension import SourceMomentData, bit_matrix, load_source_snapshot, units_from_bits, vqe_state

TRAINED_LAMBDAS: Tuple[float, ...] = (1.23737, 1.21199, 0.013695, 0.001189, 161.126)

@dataclass(frozen=True)
class SourceHuboConfig:
    mode: str = "budget_aligned"
    total_budget: float = 10_000.0
    total_units: int = 8
    bits_per_asset: int = 3
    lambda_return: float = TRAINED_LAMBDAS[0]
    lambda_variance: float = TRAINED_LAMBDAS[1]
    lambda_skewness: float = TRAINED_LAMBDAS[2]
    lambda_kurtosis: float = TRAINED_LAMBDAS[3]
    lambda_budget: float = TRAINED_LAMBDAS[4]
    risk_free_rate: float = 0.02
    layers: int = 2
    def __post_init__(self) -> None:
        if self.mode not in {"original", "budget_aligned"}: raise ValueError("mode must be 'original' or 'budget_aligned'.")
        if self.total_budget <= 0.0 or self.total_units <= 0: raise ValueError("total_budget and total_units must be positive.")
    @property
    def dollar_unit(self) -> float: return 1_150.0 if self.mode == "original" else self.total_budget / self.total_units
    @property
    def target_units(self) -> float: return self.total_budget / self.dollar_unit
    @property
    def exact_budget_representable(self) -> bool: return bool(np.isclose(self.target_units, round(self.target_units), atol=1e-12))

def _state_bitstring(units: np.ndarray, bits_per_asset: int) -> str:
    return "".join(format(int(value), f"0{bits_per_asset}b") for value in units)

def source_hubo_state_table(data: SourceMomentData, config: SourceHuboConfig) -> Dict[str, np.ndarray]:
    n_assets = len(data.tickers); n_qubits = n_assets * config.bits_per_asset; bits = bit_matrix(n_qubits); units = units_from_bits(bits, n_assets, config.bits_per_asset)
    unit_totals = units.sum(axis=1); encoded_spend = unit_totals.astype(float) * config.dollar_unit; budget_breach = np.abs(encoded_spend - config.total_budget)
    minimum_breach = float(np.min(budget_breach)); admissible = np.isclose(budget_breach, minimum_breach, atol=1e-9)
    weights = np.zeros_like(units, dtype=float); nonzero = unit_totals > 0; weights[nonzero] = units[nonzero] / unit_totals[nonzero, None]
    expected_return = weights @ data.expected_returns; variance = np.einsum("ni,ij,nj->n", weights, data.covariance, weights, optimize=True); volatility = np.sqrt(np.maximum(variance, 0.0))
    sharpe = np.zeros_like(expected_return); valid = volatility > 1e-12; sharpe[valid] = (expected_return[valid] - config.risk_free_rate) / volatility[valid]
    return_energy = -config.lambda_return * (units @ data.expected_returns)
    variance_energy = config.lambda_variance * np.einsum("ni,ij,nj->n", units, data.covariance, units, optimize=True)
    skewness_energy = -config.lambda_skewness * np.einsum("ni,nj,nk,ijk->n", units, units, units, data.co_skewness, optimize=True)
    kurtosis_energy = config.lambda_kurtosis * np.einsum("ni,nj,nk,nl,ijkl->n", units, units, units, units, data.co_kurtosis, optimize=True)
    budget_energy = config.lambda_budget * (unit_totals.astype(float) - config.target_units) ** 2
    total_energy = return_energy + variance_energy + skewness_energy + kurtosis_energy + budget_energy
    return {"bits": bits, "units": units, "unit_totals": unit_totals, "weights": weights, "encoded_spend": encoded_spend, "budget_breach": budget_breach, "admissible": admissible, "expected_return": expected_return, "variance": variance, "volatility": volatility, "sharpe_ratio": sharpe, "return_energy": return_energy, "variance_energy": variance_energy, "skewness_energy": skewness_energy, "kurtosis_energy": kurtosis_energy, "budget_energy": budget_energy, "total_energy": total_energy}

def _record(table: Mapping[str, np.ndarray], index: int, config: SourceHuboConfig) -> Dict[str, Any]:
    units = np.asarray(table["units"][index], dtype=int)
    return {"state_index": int(index), "bitstring": _state_bitstring(units, config.bits_per_asset), "units": [int(v) for v in units], "weights": [float(v) for v in table["weights"][index]], "unit_total": int(table["unit_totals"][index]), "encoded_spend": float(table["encoded_spend"][index]), "budget_breach": float(table["budget_breach"][index]), "expected_return": float(table["expected_return"][index]), "volatility": float(table["volatility"][index]), "sharpe_ratio": float(table["sharpe_ratio"][index]), "return_energy": float(table["return_energy"][index]), "variance_energy": float(table["variance_energy"][index]), "skewness_energy": float(table["skewness_energy"][index]), "kurtosis_energy": float(table["kurtosis_energy"][index]), "budget_energy": float(table["budget_energy"][index]), "total_energy": float(table["total_energy"][index]), "minimum_breach_state": bool(table["admissible"][index])}

def exact_source_hubo_report(data: SourceMomentData | None = None, config: SourceHuboConfig | None = None) -> Dict[str, Any]:
    data = load_source_snapshot() if data is None else data; config = SourceHuboConfig() if config is None else config; table = source_hubo_state_table(data, config)
    ground_index = int(np.argmin(table["total_energy"])); admissible_indices = np.flatnonzero(table["admissible"]); financial_index = int(admissible_indices[np.argmax(table["sharpe_ratio"][admissible_indices])]); admissible_energy_index = int(admissible_indices[np.argmin(table["total_energy"][admissible_indices])])
    return {"mode": config.mode, "mode_label": "Original full-tensor HUBO ($1,150 unit)" if config.mode == "original" else "Budget-aligned full-tensor HUBO ($1,250 unit)", "assets": list(data.tickers), "n_qubits": len(data.tickers) * config.bits_per_asset, "states_enumerated": int(table["bits"].shape[0]), "target_budget": float(config.total_budget), "dollar_unit": float(config.dollar_unit), "target_units": float(config.target_units), "exact_budget_representable": config.exact_budget_representable, "minimum_budget_breach": float(np.min(table["budget_breach"])), "admissible_states": int(table["admissible"].sum()), "lambdas": {"return": config.lambda_return, "variance": config.lambda_variance, "skewness": config.lambda_skewness, "kurtosis": config.lambda_kurtosis, "budget": config.lambda_budget}, "hamiltonian_ground_state": _record(table, ground_index, config), "best_admissible_hamiltonian_state": _record(table, admissible_energy_index, config), "financial_reference": _record(table, financial_index, config), "ground_state_is_minimum_breach": bool(table["admissible"][ground_index]), "ground_state_matches_financial_reference": bool(ground_index == financial_index), "mixed_tensor_terms_included": True, "claim_boundary": "Original mode reproduces the source $1,150 budget geometry and therefore cannot achieve zero budget breach. Budget-aligned mode changes only the dollar unit to $10,000 / 8 = $1,250 so eight integer units represent the target budget exactly."}

def run_source_hubo_vqe(data: SourceMomentData, config: SourceHuboConfig, *, maxiter: int = 80, shots: int = 4096, seed: int = 42, restarts: int = 2) -> Dict[str, Any]:
    if maxiter <= 0 or shots <= 0 or restarts <= 0: raise ValueError("maxiter, shots, and restarts must be positive.")
    started = perf_counter(); table = source_hubo_state_table(data, config); energies = np.asarray(table["total_energy"], dtype=float); normalized = (energies - float(energies.mean())) / max(float(energies.std()), 1e-12); n_qubits = len(data.tickers) * config.bits_per_asset; rng = np.random.default_rng(seed)
    def expectation(params: np.ndarray) -> float:
        state = vqe_state(params, n_qubits, config.layers); return float((state * state) @ normalized)
    best = None; n_params = (config.layers + 1) * n_qubits
    for _ in range(restarts):
        initial = rng.uniform(-np.pi, np.pi, size=n_params); result = minimize(expectation, initial, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.25, "tol": 1e-5})
        if best is None or float(result.fun) < float(best.fun): best = result
    assert best is not None
    state = vqe_state(best.x, n_qubits, config.layers); probabilities = state * state; probabilities /= probabilities.sum(); counts = rng.multinomial(shots, probabilities); observed = np.flatnonzero(counts > 0); admissible_observed = observed[table["admissible"][observed]]
    exact = exact_source_hubo_report(data, config); exact_index = int(exact["best_admissible_hamiltonian_state"]["state_index"]); fallback = admissible_observed.size == 0
    if fallback: selected_index = exact_index; selected_count = 0; status = "completed_with_exact_classical_fallback"
    else: selected_index = int(admissible_observed[np.argmin(table["total_energy"][admissible_observed])]); selected_count = int(counts[selected_index]); status = "completed"
    return {"status": status, "selected": _record(table, selected_index, config), "runtime_seconds": perf_counter() - started, "angles": np.asarray(best.x, dtype=float).tolist(), "metadata": {"maxiter": maxiter, "shots": shots, "restarts": restarts, "optimizer_status": str(best.message), "admissible_probability_mass": float(probabilities[table["admissible"]].sum()), "exact_admissible_optimum_probability": float(probabilities[exact_index]), "observed_unique_states": int(observed.size), "observed_admissible_states": int(admissible_observed.size), "selected_observed_count": selected_count, "exact_admissible_optimum_recovered": bool(selected_index == exact_index), "optimality_gap": float(table["total_energy"][selected_index] - table["total_energy"][exact_index]), "classical_fallback_used": fallback}}

def source_hubo_dashboard_report(*, mode: str = "budget_aligned", run_vqe: bool = False, maxiter: int = 80, shots: int = 4096, seed: int = 42) -> Dict[str, Any]:
    data = load_source_snapshot(); config = SourceHuboConfig(mode=mode); report = exact_source_hubo_report(data, config)
    if run_vqe: report["vqe"] = run_source_hubo_vqe(data, config, maxiter=maxiter, shots=shots, seed=seed)
    return report
