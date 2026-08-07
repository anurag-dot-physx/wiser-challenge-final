import numpy as np

from vanguard_copilot.audit import quadratic_model_audit
from vanguard_copilot.higher_moment_extension import load_source_snapshot
from vanguard_copilot.hubo_quadratization import (
    constraints_satisfied,
    evaluate_polynomial,
    evaluate_qubo,
    exact_quadratize_hubo,
    lift_original_bits,
    portfolio_hubo_polynomial,
)
from vanguard_copilot.source_hubo_models import SourceHuboConfig, source_hubo_state_table
from vanguard_copilot.workflow import run_challenge


def _all_bits(n):
    states = np.arange(2**n, dtype=np.uint32)
    shifts = np.arange(n, dtype=np.uint32)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def test_flagship_profiles_are_feasible_and_audited():
    for profile in ("Growth", "Balanced", "Defensive"):
        run = run_challenge(profile_name=profile, run_quantum=False)
        assert run.continuous.metrics.feasible
        assert run.discrete.metrics.feasible
        assert run.continuous.metrics.objective <= run.discrete.metrics.objective + 1e-8
        assert run.discrete.metadata["states_enumerated"] == 19448
        assert run.qubo_audit["ground_state_hard_feasible"]
        assert run.qubo_audit["ground_portfolio_matches_reduced_exact"]


def test_full_quadratic_audit_passes():
    report = quadratic_model_audit(samples=8, random_seed=7)
    assert report["all_checks_passed"]
    assert report["status"] == "passed"


def test_portfolio_hubo_polynomial_reconstructs_source_energy():
    data = load_source_snapshot()
    config = SourceHuboConfig(mode="budget_aligned")
    table = source_hubo_state_table(data, config)
    poly = portfolio_hubo_polynomial(data, config)
    expanded = np.asarray(evaluate_polynomial(poly, table["bits"]), dtype=float)
    assert max(map(len, poly)) <= 4
    assert np.max(np.abs(expanded - table["total_energy"])) < 1e-7


def test_exact_quadratization_preserves_toy_hubo():
    poly = {(): 1.25, (0,): -0.3, (1, 2): 0.7, (0, 1, 2): -3.0, (0, 1, 2, 3): 2.5}
    quad = exact_quadratize_hubo(poly, 4)
    assert quad.n_ancillas > 0
    assert quad.penalty_strength > quad.objective_range_bound
    bits = _all_bits(4)
    lifted = lift_original_bits(bits, quad)
    assert np.all(constraints_satisfied(lifted, quad))
    assert np.allclose(evaluate_polynomial(poly, bits), evaluate_qubo(quad, lifted), atol=1e-10)
    assert int(np.argmin(evaluate_polynomial(poly, bits))) == int(np.argmin(evaluate_qubo(quad, lifted)))
