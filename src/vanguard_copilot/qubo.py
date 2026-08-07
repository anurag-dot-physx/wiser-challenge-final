"""Exact-constraint QUBO construction for the reduced quadratic portfolio model."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log2
from typing import Dict, Sequence, Tuple

import numpy as np

from .model import feasible_weight_mask, objective_terms, portfolio_objective, quadratic_objectives
from .solvers import ReducedProblem, solve_reduced_exact


@dataclass(frozen=True)
class ConstraintEncoding:
    name: str
    coefficients: np.ndarray
    target: int
    relation: str
    slack_indices: Tuple[int, ...]

    def residual(self, bits: Sequence[int]) -> float:
        return float(np.asarray(bits, dtype=float) @ self.coefficients - self.target)


@dataclass(frozen=True)
class IsingModel:
    linear: np.ndarray
    quadratic_upper: np.ndarray
    offset: float

    def energy(self, spins: Sequence[int]) -> float:
        z = np.asarray(spins, dtype=float)
        if z.shape != self.linear.shape or not np.all(np.isin(z, (-1.0, 1.0))):
            raise ValueError("spins must be a vector of -1/+1 values with the correct length.")
        return float(self.offset + self.linear @ z + np.sum(self.quadratic_upper * np.outer(z, z)))


@dataclass(frozen=True)
class QuboModel:
    Q: np.ndarray
    offset: float
    allocation_map: np.ndarray
    allocation_unit_map: np.ndarray
    problem: ReducedProblem
    constraint_penalty: float
    minimum_exact_penalty: float
    financial_energy_range: float
    constraints: Tuple[ConstraintEncoding, ...]
    variable_labels: Tuple[str, ...]
    n_allocation_qubits: int
    asset_unit_caps: Tuple[int, ...]

    @property
    def n_qubits(self) -> int:
        return self.Q.shape[0]

    @property
    def n_slack_qubits(self) -> int:
        return self.n_qubits - self.n_allocation_qubits

    def _validated_bits(self, bits: Sequence[int]) -> np.ndarray:
        x = np.asarray(bits, dtype=float)
        if x.shape != (self.n_qubits,):
            raise ValueError(f"bits must have shape ({self.n_qubits},).")
        if not np.all(np.isin(x, (0.0, 1.0))):
            raise ValueError("bits must contain only 0 and 1.")
        return x

    def energy(self, bits: Sequence[int]) -> float:
        x = self._validated_bits(bits)
        return float(x @ self.Q @ x + self.offset)

    def weights(self, bits: Sequence[int]) -> np.ndarray:
        return self.allocation_map @ self._validated_bits(bits)

    def units(self, bits: Sequence[int]) -> np.ndarray:
        return self.allocation_unit_map @ self._validated_bits(bits)

    def constraint_residuals(self, bits: Sequence[int]) -> Dict[str, float]:
        x = self._validated_bits(bits)
        return {constraint.name: constraint.residual(x) for constraint in self.constraints}

    def upper_triangular_qubo(self) -> np.ndarray:
        upper = np.triu(self.Q).copy()
        indices = np.triu_indices(self.n_qubits, k=1)
        upper[indices] *= 2.0
        return upper

    def to_ising(self) -> IsingModel:
        upper = self.upper_triangular_qubo()
        diagonal = np.diag(upper)
        pair = np.triu(upper, k=1)
        linear = -0.5 * diagonal - 0.25 * (pair.sum(axis=0) + pair.sum(axis=1))
        quadratic = 0.25 * pair
        offset = float(self.offset + 0.5 * diagonal.sum() + 0.25 * pair.sum())
        return IsingModel(linear=linear, quadratic_upper=quadratic, offset=offset)


def _bounded_binary_coefficients(max_value: int) -> Tuple[int, ...]:
    if max_value < 0:
        raise ValueError("max_value must be nonnegative.")
    if max_value == 0:
        return ()
    highest_power_count = int(floor(log2(max_value)))
    coefficients = [2**index for index in range(highest_power_count)]
    remainder = max_value - sum(coefficients)
    if remainder > 0:
        coefficients.append(remainder)
    return tuple(coefficients)


def _upper_units(limit: float, step: float) -> int:
    return max(0, int(floor(limit / step + 1e-12)))


def _lower_units(limit: float, step: float) -> int:
    return max(0, int(ceil(limit / step - 1e-12)))


def _add_squared_equality_penalty(Q: np.ndarray, offset: float, coefficients: np.ndarray, target: int, strength: float) -> tuple[np.ndarray, float]:
    updated = Q + strength * np.outer(coefficients, coefficients)
    updated = updated.copy()
    updated[np.diag_indices_from(updated)] -= 2.0 * strength * target * coefficients
    return updated, float(offset + strength * target * target)


def bit_matrix(n_qubits: int, *, max_states: int = 2_000_000) -> np.ndarray:
    if n_qubits < 0 or n_qubits >= 63:
        raise ValueError("n_qubits must lie in [0, 62].")
    states = 1 << n_qubits
    if states > max_states:
        raise ValueError(f"State table would contain {states:,} rows, above max_states={max_states:,}.")
    indices = np.arange(states, dtype=np.uint64)
    shifts = np.arange(n_qubits, dtype=np.uint64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def _allocation_encoding(problem: ReducedProblem) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    data = problem.data
    profile = problem.profile
    target_units = problem.target_units
    max_by_bits = problem.encoding_max_units
    caps = [min(_upper_units(profile.asset_max, problem.step), max_by_bits, target_units) for _ in range(data.n_assets)]
    group_upper = {
        "equity": _upper_units(profile.equity_max, problem.step),
        "alternatives": _upper_units(profile.alternatives_max, problem.step),
    }
    for group, cap in group_upper.items():
        members = data.groups[group]
        if len(members) == 1:
            index = members[0]
            caps[index] = min(caps[index], cap)
    labels: list[str] = []
    columns: list[np.ndarray] = []
    for asset, cap in enumerate(caps):
        coefficients = _bounded_binary_coefficients(cap)
        for bit_number, coefficient in enumerate(coefficients):
            column = np.zeros(data.n_assets, dtype=float)
            column[asset] = coefficient
            columns.append(column)
            labels.append(f"alloc:{data.names[asset]}:b{bit_number}:u{coefficient}")
    if not columns:
        raise ValueError("Reduced encoding contains no allocation variables.")
    unit_map = np.column_stack(columns)
    return unit_map, tuple(labels), tuple(caps)


def build_qubo(problem: ReducedProblem, *, constraint_penalty: float | None = None, penalty_safety_factor: float = 1.10) -> QuboModel:
    if not np.isfinite(penalty_safety_factor) or penalty_safety_factor <= 1.0:
        raise ValueError("penalty_safety_factor must be finite and greater than one.")
    data = problem.data
    profile = problem.profile
    unit_map_alloc, allocation_labels, caps = _allocation_encoding(problem)
    n_allocation = unit_map_alloc.shape[1]
    target_units = problem.target_units
    specs: list[dict[str, object]] = [{
        "name": "fully_invested",
        "allocation_coefficients": unit_map_alloc.sum(axis=0),
        "target": target_units,
        "relation": "sum_units == target_units",
        "slack_sign": 0.0,
        "slack_max": 0,
    }]
    for group, limit in (("equity", profile.equity_max), ("alternatives", profile.alternatives_max)):
        members = data.groups[group]
        group_cap = _upper_units(limit, problem.step)
        maximum = sum(caps[index] for index in members)
        if maximum > group_cap:
            specs.append({
                "name": f"{group}_maximum",
                "allocation_coefficients": unit_map_alloc[list(members), :].sum(axis=0),
                "target": group_cap,
                "relation": f"{group}_units <= {group_cap}",
                "slack_sign": 1.0,
                "slack_max": group_cap,
            })
    defensive_members = data.groups["defensive"]
    defensive_min = _lower_units(profile.defensive_min, problem.step)
    defensive_max = sum(caps[index] for index in defensive_members)
    if defensive_max < defensive_min:
        raise ValueError("The reduced encoding cannot satisfy the defensive minimum.")
    if defensive_min > 0:
        specs.append({
            "name": "defensive_minimum",
            "allocation_coefficients": unit_map_alloc[list(defensive_members), :].sum(axis=0),
            "target": defensive_min,
            "relation": f"defensive_units >= {defensive_min}",
            "slack_sign": -1.0,
            "slack_max": defensive_max - defensive_min,
        })
    labels = list(allocation_labels)
    slack_blocks: list[tuple[int, ...]] = []
    for spec in specs:
        slack_coefficients = _bounded_binary_coefficients(int(spec["slack_max"]))
        indices: list[int] = []
        for bit_number, coefficient in enumerate(slack_coefficients):
            indices.append(len(labels))
            labels.append(f"slack:{spec['name']}:b{bit_number}:u{coefficient}")
        spec["slack_coefficients"] = slack_coefficients
        slack_blocks.append(tuple(indices))
    n_qubits = len(labels)
    unit_map = np.zeros((data.n_assets, n_qubits), dtype=float)
    unit_map[:, :n_allocation] = unit_map_alloc
    weight_map = problem.step * unit_map
    constraints: list[ConstraintEncoding] = []
    for spec, slack_indices in zip(specs, slack_blocks):
        coefficients = np.zeros(n_qubits, dtype=float)
        coefficients[:n_allocation] = np.asarray(spec["allocation_coefficients"], dtype=float)
        sign = float(spec["slack_sign"])
        for index, coefficient in zip(slack_indices, spec["slack_coefficients"]):
            coefficients[index] = sign * coefficient
        constraints.append(ConstraintEncoding(str(spec["name"]), coefficients, int(spec["target"]), str(spec["relation"]), slack_indices))
    H, g, c = objective_terms(data, profile)
    Q_financial = weight_map.T @ H @ weight_map
    Q_financial = 0.5 * (Q_financial + Q_financial.T)
    Q_financial = Q_financial.copy()
    Q_financial[np.diag_indices_from(Q_financial)] += weight_map.T @ g
    allocation_bits = bit_matrix(n_allocation)
    allocation_weights = allocation_bits @ (problem.step * unit_map_alloc).T
    financial_values = quadratic_objectives(allocation_weights, H, g, c)
    financial_min = float(np.min(financial_values))
    financial_max = float(np.max(financial_values))
    financial_range = financial_max - financial_min
    numerical_margin = max(1e-9, 0.01 * max(1.0, abs(financial_min), abs(financial_max)))
    minimum_penalty = financial_range + numerical_margin
    if constraint_penalty is None:
        penalty = penalty_safety_factor * minimum_penalty
    else:
        penalty = float(constraint_penalty)
        if not np.isfinite(penalty) or penalty <= minimum_penalty:
            raise ValueError("constraint_penalty must exceed " f"{minimum_penalty:.12g} to guarantee a feasible ground state.")
    Q = Q_financial.copy()
    offset = float(c)
    for constraint in constraints:
        Q, offset = _add_squared_equality_penalty(Q, offset, constraint.coefficients, constraint.target, penalty)
    Q = 0.5 * (Q + Q.T)
    return QuboModel(Q, offset, weight_map, unit_map, problem, penalty, minimum_penalty, financial_range, tuple(constraints), tuple(labels), n_allocation, caps)


def all_state_energies(model: QuboModel) -> Tuple[np.ndarray, np.ndarray]:
    bits = bit_matrix(model.n_qubits)
    energies = np.sum((bits @ model.Q) * bits, axis=1) + model.offset
    return bits, energies


def true_feasible_mask(model: QuboModel, bits: np.ndarray) -> np.ndarray:
    matrix = np.asarray(bits, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != model.n_qubits:
        raise ValueError("bits must have shape (n_states, n_qubits).")
    weights = matrix @ model.allocation_map.T
    return feasible_weight_mask(weights, model.problem.data, model.problem.profile, tolerance=1e-12)


def constraint_satisfied_mask(model: QuboModel, bits: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(bits, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != model.n_qubits:
        raise ValueError("bits must have shape (n_states, n_qubits).")
    result = np.ones(matrix.shape[0], dtype=bool)
    for constraint in model.constraints:
        result &= np.abs(matrix @ constraint.coefficients - constraint.target) <= tolerance
    return result


def qubo_decomposition_check(model: QuboModel, bits: Sequence[int]) -> Dict[str, object]:
    x = model._validated_bits(bits)
    weights = model.allocation_map @ x
    financial = portfolio_objective(weights, model.problem.data, model.problem.profile)
    penalties = {constraint.name: model.constraint_penalty * constraint.residual(x) ** 2 for constraint in model.constraints}
    total_penalty = float(sum(penalties.values()))
    return {"financial_objective": financial, "constraint_penalties": penalties, "constraint_penalty_total": total_penalty, "total": financial + total_penalty, "qubo_energy": model.energy(x)}


def audit_qubo(model: QuboModel) -> Dict[str, object]:
    bits, energies = all_state_energies(model)
    feasible = true_feasible_mask(model, bits)
    constraint_consistent = constraint_satisfied_mask(model, bits)
    weights = bits @ model.allocation_map.T
    H, g, c = objective_terms(model.problem.data, model.problem.profile)
    financial = quadratic_objectives(weights, H, g, c)
    ground_energy = float(np.min(energies))
    ground = np.isclose(energies, ground_energy, atol=1e-10, rtol=1e-10)
    exact = solve_reduced_exact(model.problem)
    exact_mask = feasible & np.all(np.isclose(weights, exact.weights[None, :], atol=1e-12), axis=1)
    feasible_units = np.rint(weights[feasible] / model.problem.step).astype(int)
    consistent_units = np.rint(weights[constraint_consistent] / model.problem.step).astype(int)
    unique_feasible = np.unique(feasible_units, axis=0)
    unique_consistent = np.unique(consistent_units, axis=0)
    feasible_financial_min = float(np.min(financial[feasible]))
    return {
        "states_enumerated": int(bits.shape[0]),
        "allocation_feasible_encodings": int(feasible.sum()),
        "constraint_consistent_encodings": int(constraint_consistent.sum()),
        "unique_hard_feasible_portfolios": int(unique_feasible.shape[0]),
        "unique_constraint_consistent_portfolios": int(unique_consistent.shape[0]),
        "constraint_consistent_implies_hard_feasible": bool(np.all(feasible[constraint_consistent])),
        "ground_state_encodings": int(ground.sum()),
        "ground_state_hard_feasible": bool(np.all(feasible[ground])),
        "ground_state_constraint_consistent": bool(np.all(constraint_consistent[ground])),
        "ground_portfolio_matches_reduced_exact": bool(np.all(exact_mask[ground])),
        "exact_portfolio_encodings": int(exact_mask.sum()),
        "ground_energy": ground_energy,
        "exact_financial_objective": float(exact.metrics.objective),
        "minimum_feasible_financial_objective": feasible_financial_min,
        "constraint_penalty": model.constraint_penalty,
        "minimum_exact_penalty": model.minimum_exact_penalty,
        "financial_energy_range": model.financial_energy_range,
    }
