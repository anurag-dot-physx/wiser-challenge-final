"""End-to-end audited quadratic-model workflow and report helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .model import AssetClassData, InvestorProfile, PROFILES, explain_portfolio, synthetic_asset_class_data
from .qaoa import QaoaResult, run_qaoa
from .qubo import QuboModel, audit_qubo, build_qubo
from .solvers import ReducedProblem, SolverResult, baseline_result, make_reduced_problem, solve_continuous, solve_discrete_exact, solve_reduced_exact


@dataclass(frozen=True)
class ChallengeRun:
    profile: InvestorProfile
    data: AssetClassData
    baseline: SolverResult
    continuous: SolverResult
    discrete: SolverResult
    reduced_problem: ReducedProblem | None
    reduced_exact: SolverResult | None
    qubo: QuboModel | None
    qubo_audit: Mapping[str, object] | None
    qaoa: QaoaResult | None
    reduced_error: str | None

    @property
    def reduced_available(self) -> bool:
        return self.reduced_problem is not None and self.reduced_exact is not None and self.qubo is not None and self.qubo_audit is not None and self.reduced_error is None


def run_challenge(*, profile_name: str = "Balanced", profile_overrides: Mapping[str, float] | None = None, discrete_step: float = 0.10, run_quantum: bool = True, qaoa_p: int = 1, qaoa_maxiter: int = 60, qaoa_shots: int = 4096, seed: int = 42) -> ChallengeRun:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choose from {tuple(PROFILES)}.")
    profile = PROFILES[profile_name]
    if profile_overrides:
        profile = profile.with_overrides(**dict(profile_overrides))
    data = synthetic_asset_class_data()
    baseline = baseline_result(data, profile)
    continuous = solve_continuous(data, profile)
    if continuous.status != "optimal":
        raise RuntimeError(f"Continuous convex QP failed: {continuous.metadata['message']}")
    discrete = solve_discrete_exact(data, profile, step=discrete_step)
    if continuous.metrics.objective > discrete.metrics.objective + 1e-8:
        raise RuntimeError("Continuous optimum is worse than the discrete-grid optimum; solver audit failed.")
    reduced_problem = None; reduced_exact = None; qubo = None; qubo_audit_result = None; qaoa = None; reduced_error = None
    try:
        reduced_problem = make_reduced_problem(data, profile)
        reduced_exact = solve_reduced_exact(reduced_problem)
        qubo = build_qubo(reduced_problem)
        qubo_audit_result = audit_qubo(qubo)
        required = ("constraint_consistent_implies_hard_feasible", "ground_state_hard_feasible", "ground_state_constraint_consistent", "ground_portfolio_matches_reduced_exact")
        if not all(bool(qubo_audit_result[name]) for name in required):
            raise RuntimeError("The reduced QUBO failed its exact exhaustive audit.")
        if run_quantum:
            qaoa = run_qaoa(qubo, p=qaoa_p, maxiter=qaoa_maxiter, shots=qaoa_shots, seed=seed, model_audit=qubo_audit_result)
    except (ValueError, RuntimeError) as exc:
        reduced_problem = None; reduced_exact = None; qubo = None; qubo_audit_result = None; qaoa = None
        reduced_error = f"{type(exc).__name__}: {exc}"
    return ChallengeRun(profile, data, baseline, continuous, discrete, reduced_problem, reduced_exact, qubo, qubo_audit_result, qaoa, reduced_error)


def _metrics_dict(metrics: Any) -> Dict[str, Any]:
    result = asdict(metrics)
    result["hard_breaches"] = list(result["hard_breaches"])
    result["scenario_losses"] = list(result["scenario_losses"])
    result["feasible"] = not result["hard_breaches"]
    return result


def _solver_dict(result: SolverResult, names: tuple[str, ...]) -> Dict[str, Any]:
    return {"method": result.method, "status": result.status, "weights": {name: float(weight) for name, weight in zip(names, result.weights)}, "metrics": _metrics_dict(result.metrics), "runtime_seconds": float(result.runtime_seconds), "metadata": dict(result.metadata)}


def _qaoa_dict(result: QaoaResult, names: tuple[str, ...]) -> Dict[str, Any]:
    return {"method": result.method, "status": result.status, "weights": {name: float(weight) for name, weight in zip(names, result.selected_weights)}, "metrics": _metrics_dict(result.metrics), "runtime_seconds": float(result.runtime_seconds), "angles": result.angles.tolist(), "metadata": dict(result.metadata)}


def challenge_report(run: ChallengeRun) -> Dict[str, Any]:
    production_names = run.data.names
    report = {
        "scope_note": "The eight-asset production model and reduced quantum model are separate benchmarks. Quantum results are compared only with the reduced exact classical reference.",
        "profile": asdict(run.profile),
        "data_source": "deterministic synthetic/anonymized asset-class assumptions",
        "quadratic_model": {
            "objective": "risk - expected return - income + calibrated quadratic rebalancing proxy + mean squared scenario loss",
            "continuous_problem_convex": True,
            "turnover_convention": "one-way turnover = 0.5 * sum(abs(new_weight - current_weight))",
            "estimated_cost_convention": "sum(transaction_cost_rate * gross absolute weight traded)",
        },
        "production_model": {
            "baseline": _solver_dict(run.baseline, production_names),
            "continuous": _solver_dict(run.continuous, production_names),
            "discrete_exact": _solver_dict(run.discrete, production_names),
            "continuous_to_discrete_objective_gap": float(run.discrete.metrics.objective - run.continuous.metrics.objective),
            "explanation": list(explain_portfolio(run.discrete.weights, run.data, run.profile, run.baseline.weights)),
        },
    }
    if not run.reduced_available:
        report["reduced_quantum_model"] = {"status": "unavailable_for_selected_guardrails", "reason": run.reduced_error, "allocation_step": 0.125, "claim_boundary": "The production portfolio remains valid; only the coarse reduced quantum encoding is unavailable."}
        return report
    assert run.reduced_problem is not None and run.reduced_exact is not None and run.qubo is not None and run.qubo_audit is not None
    reduced_names = run.reduced_problem.data.names
    reduced_report = {
        "status": "available",
        "asset_classes": list(reduced_names),
        "requested_bits_per_asset": run.reduced_problem.bits_per_asset,
        "allocation_step": run.reduced_problem.step,
        "n_qubits": run.qubo.n_qubits,
        "allocation_qubits": run.qubo.n_allocation_qubits,
        "slack_qubits": run.qubo.n_slack_qubits,
        "asset_unit_caps": list(run.qubo.asset_unit_caps),
        "exact_classical": _solver_dict(run.reduced_exact, reduced_names),
        "qubo": {"constraint_penalty": run.qubo.constraint_penalty, "minimum_exact_penalty": run.qubo.minimum_exact_penalty, "financial_energy_range": run.qubo.financial_energy_range, "matrix_shape": list(run.qubo.Q.shape), "constraints": [{"name": c.name, "relation": c.relation, "target": c.target, "slack_qubits": len(c.slack_indices)} for c in run.qubo.constraints], "audit": dict(run.qubo_audit), "convention": "symmetric x.T @ Q @ x + offset"},
    }
    if run.qaoa is not None:
        reduced_report["qaoa"] = _qaoa_dict(run.qaoa, reduced_names)
        reduced_report["qaoa_optimality_gap"] = float(run.qaoa.metrics.objective - run.reduced_exact.metrics.objective)
    report["reduced_quantum_model"] = reduced_report
    return report


def comparison_rows(run: ChallengeRun) -> list[Dict[str, Any]]:
    rows = []
    for label, result in (("Current portfolio", run.baseline), ("Continuous convex QP", run.continuous), ("Discrete exact grid", run.discrete)):
        m = result.metrics
        rows.append({"Method": label, "Expected return": m.expected_return, "Volatility": m.volatility, "Risk-adjusted ratio": m.sharpe_like, "Income yield": m.income_yield, "One-way turnover": m.turnover, "Gross turnover": m.gross_turnover, "Estimated cost": m.estimated_cost, "Worst scenario loss": m.worst_scenario_loss, "Objective": m.objective, "Hard breaches": len(m.hard_breaches), "Runtime (s)": result.runtime_seconds})
    return rows
