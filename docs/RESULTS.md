# Results and findings

## Verified flagship audit

The final development audit of the flagship quadratic implementation passed all mathematical and numerical checks for the Growth, Balanced, and Defensive profiles.

| Profile | Max objective-form error | Min eigenvalue of H | Continuous→10% grid objective gap | Grid states | Reduced QUBO variables | Unique feasible reduced portfolios |
|---|---:|---:|---:|---:|---:|---:|
| Growth | 5.551e-17 | 1.010e-02 | 0.00081799 | 19,448 | 13 | 108 |
| Balanced | 6.939e-17 | 2.185e-02 | 0.00066947 | 19,448 | 15 | 66 |
| Defensive | 4.857e-17 | 3.365e-02 | 0.00140581 | 19,448 | 11 | 2 |

The minimum Hessian eigenvalue is positive for all three profiles, so the reported continuous problem is strictly convex for these inputs. The complete 10% grid is enumerated rather than sampled. The continuous solution is therefore a valid lower bound on the grid objective, and the positive gaps above quantify the discretization cost.

The turnover-sensitivity audit changes one-way turnover from **40%** at zero rebalancing weight to **15%** at a rebalancing weight of two. This demonstrates that the cost/rebalancing control produces a material change rather than being a cosmetic dashboard setting.

Every flagship continuous and exact-grid solution passed the original hard-constraint validation. The reduced QUBO audit additionally verifies that its constraint-consistent state space equals the hard-feasible reduced portfolio set and that every QUBO ground portfolio matches the exact reduced financial optimum.

## What these results establish

They establish implementation correctness for the stated synthetic/anonymized problem: objective decomposition, convexity, hard-feasibility, exact discrete optimality on the declared grid, exact reduced-QUBO encoding, and sensitivity to investor preferences.

They do **not** establish economic predictiveness or quantum advantage. Those are intentionally outside the claim boundary of this prototype.

## Higher-moment extension

The higher-moment appendix provides a controlled environment for studying return, covariance, co-skewness, and co-kurtosis in one binary Hamiltonian. Exact enumeration supplies three distinct references: the financial maximum-Sharpe portfolio, the exact feasible learned-HUBO ground state, and the unrestricted Hamiltonian ground state. This makes it possible to distinguish financial-model error from quantum-solver error.

The coefficient-training pipeline uses one cached CMA-ES trajectory and selects its generation depth on validation sectors. The selected coefficients are then frozen before held-out evaluation. A second checkpoint sweep selects an efficient final VQE optimizer budget on the fixed Hamiltonian.

## Exact quadratization finding

The quadratization module verifies two stronger properties than a simple final-state comparison:

1. the explicit pseudo-Boolean expansion reconstructs the existing full-tensor HUBO energy; and
2. on every original bit assignment, the ancilla-consistent QUBO energy equals the HUBO energy.

For small expanded instances the runner can also enumerate the complete original+ancilla QUBO state space. For larger expansions, the exact lifted-subspace check remains available and the certified Rosenberg penalty prevents inconsistent ancillas from becoming the global optimum.

## Reproduce the canonical audit

```bash
python src/audit_quadratic_model.py
```

The command writes `output/quadratic_model_audit.json`. The submission-focused regression suite is run with `pytest -q`.
