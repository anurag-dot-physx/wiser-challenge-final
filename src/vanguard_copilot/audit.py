"""Programmatic audit of the flagship convex QP, exact grid, and reduced QUBO."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .model import (
    PROFILES,
    objective_breakdown,
    objective_terms,
    quadratic_objectives,
    synthetic_asset_class_data,
)
from .solvers import solve_discrete_exact
from .workflow import run_challenge


def quadratic_model_audit(
    *,
    random_seed: int = 17,
    samples: int = 64,
) -> Dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive.")
    data = synthetic_asset_class_data()
    rng = np.random.default_rng(random_seed)
    profile_reports: Dict[str, Any] = {}
    all_checks = True

    for name, profile in PROFILES.items():
        H, g, c = objective_terms(data, profile)
        test_weights = rng.dirichlet(
            np.ones(data.n_assets),
            size=samples,
        )
        matrix_values = quadratic_objectives(
            test_weights,
            H,
            g,
            c,
        )
        explicit_values = np.array(
            [
                objective_breakdown(weight, data, profile).total
                for weight in test_weights
            ]
        )
        matrix_error = float(
            np.max(np.abs(matrix_values - explicit_values))
        )
        run = run_challenge(
            profile_name=name,
            run_quantum=False,
        )
        checks = {
            "objective_matrix_matches_decomposition": (
                matrix_error <= 1e-11
            ),
            "quadratic_hessian_psd": (
                float(np.min(np.linalg.eigvalsh(H))) >= -1e-10
            ),
            "continuous_solution_feasible": (
                run.continuous.metrics.feasible
            ),
            "discrete_solution_feasible": (
                run.discrete.metrics.feasible
            ),
            "continuous_lower_bound_respected": (
                run.continuous.metrics.objective
                <= run.discrete.metrics.objective + 1e-8
            ),
            "qubo_constraint_subspace_exact": (
                run.qubo_audit[
                    "constraint_consistent_implies_hard_feasible"
                ]
                and run.qubo_audit[
                    "unique_constraint_consistent_portfolios"
                ]
                == run.qubo_audit[
                    "unique_hard_feasible_portfolios"
                ]
            ),
            "qubo_ground_feasible": (
                run.qubo_audit["ground_state_hard_feasible"]
            ),
            "qubo_ground_matches_reduced_exact": (
                run.qubo_audit[
                    "ground_portfolio_matches_reduced_exact"
                ]
            ),
        }
        all_checks &= all(checks.values())
        profile_reports[name] = {
            "checks": checks,
            "maximum_objective_form_error": matrix_error,
            "minimum_hessian_eigenvalue": float(
                np.min(np.linalg.eigvalsh(H))
            ),
            "continuous_objective": float(
                run.continuous.metrics.objective
            ),
            "discrete_objective": float(
                run.discrete.metrics.objective
            ),
            "continuous_to_discrete_gap": float(
                run.discrete.metrics.objective
                - run.continuous.metrics.objective
            ),
            "discrete_states": int(
                run.discrete.metadata["states_enumerated"]
            ),
            "discrete_feasible_states": int(
                run.discrete.metadata["feasible_states"]
            ),
            "production_one_way_turnover": float(
                run.discrete.metrics.turnover
            ),
            "reduced_qubits": int(run.qubo.n_qubits),
            "reduced_allocation_qubits": int(
                run.qubo.n_allocation_qubits
            ),
            "reduced_slack_qubits": int(
                run.qubo.n_slack_qubits
            ),
            "reduced_unique_feasible_portfolios": int(
                run.qubo_audit[
                    "unique_hard_feasible_portfolios"
                ]
            ),
            "constraint_penalty": float(
                run.qubo.constraint_penalty
            ),
            "minimum_exact_penalty": float(
                run.qubo.minimum_exact_penalty
            ),
            "runtimes": {
                "continuous_seconds": float(
                    run.continuous.runtime_seconds
                ),
                "discrete_seconds": float(
                    run.discrete.runtime_seconds
                ),
                "reduced_exact_seconds": float(
                    run.reduced_exact.runtime_seconds
                ),
            },
        }

    balanced = PROFILES["Balanced"]
    low_turnover = solve_discrete_exact(
        data,
        balanced.with_overrides(turnover_weight=0.0),
        step=0.10,
    )
    high_turnover = solve_discrete_exact(
        data,
        balanced.with_overrides(turnover_weight=2.0),
        step=0.10,
    )
    sensitivity_check = (
        high_turnover.metrics.turnover
        < low_turnover.metrics.turnover
    )
    all_checks &= sensitivity_check

    return {
        "status": "passed" if all_checks else "failed",
        "all_checks_passed": bool(all_checks),
        "profiles": profile_reports,
        "tunable_goal_checks": {
            "balanced_turnover_preference_material": bool(
                sensitivity_check
            ),
            "turnover_at_weight_0": float(
                low_turnover.metrics.turnover
            ),
            "turnover_at_weight_2": float(
                high_turnover.metrics.turnover
            ),
        },
        "objective_definition": {
            "matrix_form": "J(w) = w.T H w + g.T w + c",
            "components": [
                "negative expected-return reward",
                "variance penalty",
                "negative income reward",
                "cost-tilted quadratic rebalancing penalty",
                "mean squared scenario-loss penalty",
            ],
            "profile_names": list(PROFILES),
        },
    }
