"""Validated quadratic portfolio model and reporting metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


ASSET_NAMES: Tuple[str, ...] = (
    "US Equity",
    "International Equity",
    "Emerging Markets",
    "Government Bonds",
    "Corporate Bonds",
    "Commodities",
    "Real Estate",
    "Cash",
)
REQUIRED_GROUPS: Tuple[str, ...] = ("equity", "defensive", "alternatives")


def _readonly_float_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AssetClassData:
    names: Tuple[str, ...]
    expected_returns: np.ndarray
    covariance: np.ndarray
    income_yields: np.ndarray
    transaction_costs: np.ndarray
    current_weights: np.ndarray
    scenario_losses: np.ndarray
    scenario_names: Tuple[str, ...]
    groups: Mapping[str, Tuple[int, ...]]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Asset names must be nonempty and unique.")
        n = len(names)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "expected_returns", _readonly_float_array(self.expected_returns, name="expected_returns", shape=(n,)))
        object.__setattr__(self, "covariance", _readonly_float_array(self.covariance, name="covariance", shape=(n, n)))
        object.__setattr__(self, "income_yields", _readonly_float_array(self.income_yields, name="income_yields", shape=(n,)))
        object.__setattr__(self, "transaction_costs", _readonly_float_array(self.transaction_costs, name="transaction_costs", shape=(n,)))
        object.__setattr__(self, "current_weights", _readonly_float_array(self.current_weights, name="current_weights", shape=(n,)))
        scenario = _readonly_float_array(self.scenario_losses, name="scenario_losses")
        if scenario.ndim != 2 or scenario.shape[1] != n or scenario.shape[0] == 0:
            raise ValueError("scenario_losses must have shape (n_scenarios, n_assets) with n_scenarios > 0.")
        scenario_names = tuple(str(name) for name in self.scenario_names)
        if len(scenario_names) != scenario.shape[0] or len(set(scenario_names)) != len(scenario_names):
            raise ValueError("scenario_names must be unique and match scenario_losses rows.")
        object.__setattr__(self, "scenario_losses", scenario)
        object.__setattr__(self, "scenario_names", scenario_names)
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("covariance must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        eig_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -eig_tolerance:
            raise ValueError("covariance must be positive semidefinite.")
        if np.any(np.diag(self.covariance) < -1e-14):
            raise ValueError("covariance diagonal entries must be nonnegative.")
        if np.any(self.transaction_costs < 0.0):
            raise ValueError("transaction_costs must be nonnegative.")
        if np.any(self.current_weights < -1e-12) or not np.isclose(self.current_weights.sum(), 1.0, atol=1e-12):
            raise ValueError("current_weights must be nonnegative and sum to one.")
        groups = {str(name): tuple(int(index) for index in indices) for name, indices in self.groups.items()}
        if set(groups) != set(REQUIRED_GROUPS):
            raise ValueError(f"groups must contain exactly {REQUIRED_GROUPS}.")
        seen: set[int] = set()
        for name in REQUIRED_GROUPS:
            indices = groups[name]
            if not indices or len(set(indices)) != len(indices):
                raise ValueError(f"Group {name!r} must contain unique asset indices.")
            if any(index < 0 or index >= n for index in indices):
                raise ValueError(f"Group {name!r} contains an invalid asset index.")
            overlap = seen.intersection(indices)
            if overlap:
                raise ValueError(f"Asset indices {sorted(overlap)} occur in multiple groups.")
            seen.update(indices)
        if seen != set(range(n)):
            missing = sorted(set(range(n)) - seen)
            raise ValueError("Every asset must belong to exactly one group; " f"missing indices: {missing}.")
        object.__setattr__(self, "groups", MappingProxyType(groups))

    @property
    def n_assets(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class InvestorProfile:
    name: str
    return_weight: float
    risk_weight: float
    income_weight: float
    turnover_weight: float
    drawdown_weight: float
    equity_max: float
    defensive_min: float
    alternatives_max: float
    asset_max: float

    def __post_init__(self) -> None:
        objective_values = (self.return_weight, self.risk_weight, self.income_weight, self.turnover_weight, self.drawdown_weight)
        limits = (self.equity_max, self.defensive_min, self.alternatives_max, self.asset_max)
        if not all(np.isfinite(value) for value in (*objective_values, *limits)):
            raise ValueError("Profile parameters must be finite.")
        if any(value < 0.0 for value in objective_values):
            raise ValueError("Objective weights must be nonnegative.")
        if not all(0.0 <= value <= 1.0 for value in limits):
            raise ValueError("Allocation guardrails must lie in [0, 1].")
        if self.asset_max <= 0.0:
            raise ValueError("asset_max must be positive.")

    def with_overrides(self, **changes: float) -> "InvestorProfile":
        return replace(self, **changes)


PROFILES: Dict[str, InvestorProfile] = {
    "Growth": InvestorProfile("Growth", 3.2, 1.0, 0.15, 0.30, 0.45, 0.80, 0.15, 0.25, 0.45),
    "Balanced": InvestorProfile("Balanced", 2.1, 1.8, 0.45, 0.65, 0.90, 0.65, 0.25, 0.20, 0.40),
    "Defensive": InvestorProfile("Defensive", 1.2, 3.0, 0.90, 1.00, 1.70, 0.45, 0.40, 0.15, 0.35),
}


@dataclass(frozen=True)
class ObjectiveBreakdown:
    return_contribution: float
    risk_penalty: float
    income_contribution: float
    rebalancing_penalty: float
    scenario_penalty: float
    total: float


@dataclass(frozen=True)
class PortfolioMetrics:
    expected_return: float
    volatility: float
    sharpe_like: float
    income_yield: float
    turnover: float
    gross_turnover: float
    estimated_cost: float
    scenario_losses: Tuple[float, ...]
    worst_scenario_loss: float
    objective: float
    objective_breakdown: ObjectiveBreakdown
    hard_breaches: Tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.hard_breaches


def synthetic_asset_class_data() -> AssetClassData:
    expected_returns = np.array([0.082, 0.076, 0.094, 0.035, 0.051, 0.046, 0.063, 0.020])
    vol = np.array([0.180, 0.195, 0.255, 0.060, 0.095, 0.205, 0.170, 0.010])
    corr = np.array([
        [1.00, 0.78, 0.68, -0.18, 0.16, 0.05, 0.55, 0.00],
        [0.78, 1.00, 0.72, -0.12, 0.18, 0.10, 0.50, 0.00],
        [0.68, 0.72, 1.00, -0.08, 0.20, 0.16, 0.44, 0.00],
        [-0.18, -0.12, -0.08, 1.00, 0.55, -0.02, -0.10, 0.05],
        [0.16, 0.18, 0.20, 0.55, 1.00, 0.04, 0.14, 0.04],
        [0.05, 0.10, 0.16, -0.02, 0.04, 1.00, 0.18, 0.00],
        [0.55, 0.50, 0.44, -0.10, 0.14, 0.18, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.05, 0.04, 0.00, 0.00, 1.00],
    ])
    covariance = np.outer(vol, vol) * corr
    return AssetClassData(
        names=ASSET_NAMES,
        expected_returns=expected_returns,
        covariance=covariance,
        income_yields=np.array([0.018, 0.025, 0.030, 0.033, 0.046, 0.010, 0.042, 0.020]),
        transaction_costs=np.array([0.0010, 0.0013, 0.0022, 0.0005, 0.0008, 0.0025, 0.0018, 0.0001]),
        current_weights=np.array([0.30, 0.15, 0.05, 0.20, 0.10, 0.05, 0.10, 0.05]),
        scenario_losses=np.array([
            [0.30, 0.33, 0.42, -0.07, 0.09, -0.05, 0.26, 0.00],
            [0.11, 0.13, 0.17, 0.10, 0.08, -0.12, 0.12, 0.01],
            [0.05, 0.06, 0.08, 0.15, 0.12, 0.02, 0.07, 0.00],
            [0.19, 0.22, 0.28, -0.04, 0.06, 0.07, 0.20, 0.00],
        ]),
        scenario_names=("Equity selloff", "Inflation shock", "Rate shock", "Recession"),
        groups={"equity": (0, 1, 2), "defensive": (3, 4, 7), "alternatives": (5, 6)},
    )


def _validated_weights(weights: Sequence[float], data: AssetClassData) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (data.n_assets,):
        raise ValueError(f"weights must have shape ({data.n_assets},).")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    return w


def rebalancing_matrix(data: AssetClassData) -> np.ndarray:
    positive_costs = data.transaction_costs[data.transaction_costs > 0.0]
    cost_reference = float(np.median(positive_costs)) if positive_costs.size else 1.0
    variance_reference = max(float(np.median(np.diag(data.covariance))), 1e-8)
    relative_cost = 1.0 + data.transaction_costs / cost_reference
    return variance_reference * np.diag(relative_cost)


def objective_terms(data: AssetClassData, profile: InvestorProfile) -> Tuple[np.ndarray, np.ndarray, float]:
    rebalance = rebalancing_matrix(data)
    scenario_quadratic = data.scenario_losses.T @ data.scenario_losses / data.scenario_losses.shape[0]
    H = profile.risk_weight * data.covariance + profile.turnover_weight * rebalance + profile.drawdown_weight * scenario_quadratic
    H = 0.5 * (H + H.T)
    g = -profile.return_weight * data.expected_returns - profile.income_weight * data.income_yields - 2.0 * profile.turnover_weight * (rebalance @ data.current_weights)
    c = float(profile.turnover_weight * data.current_weights @ rebalance @ data.current_weights)
    return H, g, c


def objective_breakdown(weights: Sequence[float], data: AssetClassData, profile: InvestorProfile) -> ObjectiveBreakdown:
    w = _validated_weights(weights, data)
    delta = w - data.current_weights
    scenario = data.scenario_losses @ w
    return_part = -profile.return_weight * float(w @ data.expected_returns)
    risk_part = profile.risk_weight * float(w @ data.covariance @ w)
    income_part = -profile.income_weight * float(w @ data.income_yields)
    rebalance = rebalancing_matrix(data)
    rebalance_part = profile.turnover_weight * float(delta @ rebalance @ delta)
    scenario_part = profile.drawdown_weight * float(np.mean(scenario * scenario))
    total = return_part + risk_part + income_part + rebalance_part + scenario_part
    return ObjectiveBreakdown(return_part, risk_part, income_part, rebalance_part, scenario_part, total)


def hard_guardrail_breaches(weights: Sequence[float], data: AssetClassData, profile: InvestorProfile, *, tolerance: float = 1e-8) -> Tuple[str, ...]:
    w = np.asarray(weights, dtype=float)
    if w.shape != (data.n_assets,):
        return ("dimension mismatch",)
    if not np.all(np.isfinite(w)):
        return ("non-finite allocation",)
    breaches: list[str] = []
    if np.any(w < -tolerance): breaches.append("negative allocation")
    if abs(float(w.sum()) - 1.0) > tolerance: breaches.append("weights do not sum to 100%")
    if np.any(w > profile.asset_max + tolerance): breaches.append("single-asset maximum exceeded")
    equity = float(w[list(data.groups["equity"])].sum())
    defensive = float(w[list(data.groups["defensive"])].sum())
    alternatives = float(w[list(data.groups["alternatives"])].sum())
    if equity > profile.equity_max + tolerance: breaches.append("equity maximum exceeded")
    if defensive < profile.defensive_min - tolerance: breaches.append("defensive minimum not met")
    if alternatives > profile.alternatives_max + tolerance: breaches.append("alternatives maximum exceeded")
    return tuple(breaches)


def feasible_weight_mask(weights: np.ndarray, data: AssetClassData, profile: InvestorProfile, *, tolerance: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != data.n_assets:
        raise ValueError("weights must have shape (n_portfolios, n_assets).")
    mask = np.all(np.isfinite(matrix), axis=1)
    mask &= np.all(matrix >= -tolerance, axis=1)
    mask &= np.abs(matrix.sum(axis=1) - 1.0) <= tolerance
    mask &= np.all(matrix <= profile.asset_max + tolerance, axis=1)
    mask &= matrix[:, list(data.groups["equity"])].sum(axis=1) <= profile.equity_max + tolerance
    mask &= matrix[:, list(data.groups["defensive"])].sum(axis=1) >= profile.defensive_min - tolerance
    mask &= matrix[:, list(data.groups["alternatives"])].sum(axis=1) <= profile.alternatives_max + tolerance
    return mask


def portfolio_objective(weights: Sequence[float], data: AssetClassData, profile: InvestorProfile) -> float:
    return objective_breakdown(weights, data, profile).total


def quadratic_objectives(weights: np.ndarray, H: np.ndarray, g: np.ndarray, c: float) -> np.ndarray:
    matrix = np.asarray(weights, dtype=float)
    return np.einsum("ni,ij,nj->n", matrix, H, matrix, optimize=True) + matrix @ g + c


def portfolio_metrics(weights: Sequence[float], data: AssetClassData, profile: InvestorProfile) -> PortfolioMetrics:
    w = _validated_weights(weights, data)
    expected_return = float(w @ data.expected_returns)
    variance = float(w @ data.covariance @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_like = float((expected_return - 0.02) / max(volatility, 1e-12))
    income_yield = float(w @ data.income_yields)
    delta_abs = np.abs(w - data.current_weights)
    gross_turnover = float(delta_abs.sum())
    one_way_turnover = 0.5 * gross_turnover
    estimated_cost = float(delta_abs @ data.transaction_costs)
    losses = tuple(float(value) for value in data.scenario_losses @ w)
    breakdown = objective_breakdown(w, data, profile)
    H, g, c = objective_terms(data, profile)
    matrix_total = float(w @ H @ w + g @ w + c)
    if not np.isclose(matrix_total, breakdown.total, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Quadratic matrix form and explicit objective decomposition disagree.")
    return PortfolioMetrics(expected_return, volatility, sharpe_like, income_yield, one_way_turnover, gross_turnover, estimated_cost, losses, float(max(losses)), breakdown.total, breakdown, hard_guardrail_breaches(w, data, profile))


def explain_portfolio(weights: Sequence[float], data: AssetClassData, profile: InvestorProfile, baseline_weights: Sequence[float] | None = None) -> Tuple[str, ...]:
    w = _validated_weights(weights, data)
    baseline = data.current_weights if baseline_weights is None else _validated_weights(baseline_weights, data)
    metrics = portfolio_metrics(w, data, profile)
    changes = w - baseline
    top = np.argsort(w)[::-1][:3]
    increased = np.argsort(changes)[::-1][:2]
    decreased = np.argsort(changes)[:2]
    equity = float(w[list(data.groups["equity"])].sum())
    defensive = float(w[list(data.groups["defensive"])].sum())
    alternatives = float(w[list(data.groups["alternatives"])].sum())
    reasons = [
        f"The {profile.name.lower()} profile places its largest weights in " + ", ".join(f"{data.names[index]} ({100*w[index]:.0f}%)" for index in top) + ".",
        "Largest increases versus the starting portfolio: " + ", ".join(f"{data.names[index]} ({100*changes[index]:+.0f} pp)" for index in increased) + ".",
        "Largest reductions versus the starting portfolio: " + ", ".join(f"{data.names[index]} ({100*changes[index]:+.0f} pp)" for index in decreased) + ".",
        f"Guardrails: equity {100*equity:.0f}% <= {100*profile.equity_max:.0f}%, defensive {100*defensive:.0f}% >= {100*profile.defensive_min:.0f}%, and alternatives {100*alternatives:.0f}% <= {100*profile.alternatives_max:.0f}%.",
        f"Expected return is {100*metrics.expected_return:.2f}%, volatility {100*metrics.volatility:.2f}%, one-way turnover {100*metrics.turnover:.2f}%, and worst modeled scenario loss {100*metrics.worst_scenario_loss:.2f}%.",
    ]
    reasons.append("All hard constraints are satisfied after validation." if metrics.feasible else "Hard-constraint warning: " + "; ".join(metrics.hard_breaches) + ".")
    return tuple(reasons)


def objective_breakdown_dict(value: ObjectiveBreakdown) -> Dict[str, float]:
    return {key: float(number) for key, number in asdict(value).items()}
