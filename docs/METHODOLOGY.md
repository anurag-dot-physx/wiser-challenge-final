# Methodology

## 1. Challenge framing

The primary task is constrained multi-asset portfolio construction: recommend an allocation that improves expected utility while satisfying hard investment guardrails and exposing transparent growth, income, drawdown, and implementation-cost controls.

The submission deliberately separates a **flagship quadratic challenge solution** from a **higher-moment research extension**. This avoids claiming that an exploratory quantum benchmark produced the eight-asset production recommendation.

## 2. Flagship eight-asset quadratic model

Let `w` be the allocation vector. The production objective is

\[
J(w)= -\lambda_\mu\mu^T w
+\lambda_r w^T\Sigma w
-\lambda_y y^T w
+\lambda_T(w-w_0)^T R(w-w_0)
+\lambda_s\frac1S\sum_{s=1}^S(\ell_s^Tw)^2.
\]

Here `mu` is expected return, `Sigma` covariance, `y` income yield, `w0` the current portfolio, `R` a positive-semidefinite cost-tilted rebalancing matrix, and `ell_s` scenario-loss vectors. The model is convex because the full quadratic Hessian is positive semidefinite.

Hard constraints enforce full investment, long-only weights, a maximum single-asset weight, an equity maximum, a defensive minimum, and an alternatives maximum.

Three profiles—Growth, Balanced, Defensive—change objective weights and guardrails while preserving the same mathematical structure.

## 3. Classical validation

The continuous model is solved as a convex constrained quadratic program. A 10% production grid is also solved by exhaustive vectorized enumeration. With eight assets and ten allocation units, the complete grid contains

\[
\binom{17}{7}=19{,}448
\]

fully invested portfolios. The continuous optimum must be a lower bound on the discrete optimum; every reported allocation is independently re-checked against the original guardrails.

## 4. Reduced QUBO and QAOA benchmark

A five-sleeve representative model is encoded on a 12.5% grid. Bounded binary integer blocks encode allocations, while nonredundant group inequalities are converted to quadratic equalities using binary slack variables:

\[
E(x,s)=J(Ax)+P\sum_k(c_k^Tx+d_k^Ts_k-t_k)^2.
\]

The constraint penalty is selected above the complete financial-energy range, so an infeasible state cannot beat a feasible state at the QUBO ground energy. The entire reduced QUBO is exhaustively audited. Depending on the investor profile it requires 11–15 binary variables.

QAOA is then evaluated only against this reduced exact reference. Finite-shot samples are post-selected only among hard-feasible portfolios; any classical fallback is explicitly reported.

## 5. Higher-moment HUBO extension

The research extension adds co-skewness and co-kurtosis:

\[
E(m)=
-\lambda_\mu\mu^Tm
+\lambda_\Sigma m^T\Sigma m
-\lambda_S\sum_{ijk}S_{ijk}m_im_jm_k
+\lambda_K\sum_{ijkl}K_{ijkl}m_im_jm_km_l
+\lambda_B\left(\sum_i m_i-8\right)^2.
\]

Three bits per asset encode integer units `m_i` in `[0,7]`. In the budget-aligned variant, one unit is `$10,000/8 = $1,250`, so exactly eight units represents the target budget with zero encoded breach.

The five coefficients are calibrated using a compact full-covariance CMA-ES in `log10(lambda)` space. All state-dependent return/variance/skewness/kurtosis/budget features are cached once; each CMA population is then evaluated with one matrix multiplication. VQE is not inside the coefficient-training loop.

## 6. Validation-selected training depth

One CMA-ES trajectory is run to the largest requested generation checkpoint. The globally best coefficient vector is snapshotted at checkpoints such as `5, 10, 20, 30, 50, ...`. Each snapshot is scored on five validation sector universes. Selection can prioritize highest validation Sharpe, highest return, lowest volatility, or the fastest checkpoint within a tolerance of the best validation Sharpe.

The last five sector universes are held out until after `G*` and the coefficient vector are fixed.

## 7. Final VQE checkpoint selection

For the selected fixed Hamiltonian, VQE is run at optimizer-evaluation checkpoints such as `20, 40, 60, 80, 120, ...`. The selection rule first prefers the smallest evaluation budget that recovers the exact feasible HUBO ground state without fallback. If no checkpoint recovers it, the checkpoint with the smallest exact Hamiltonian optimality gap is selected, with Sharpe error and runtime as tie-breakers.

## 8. Exact HUBO → QUBO quadratization

The full HUBO can also be reduced exactly to a QUBO. Cubic and quartic pseudo-Boolean monomials are replaced with reusable product ancillas. For `y = ab`, the Rosenberg constraint

\[
P(ab-2ay-2by+3y)
\]

is zero exactly when the product identity holds. The implementation chooses `P` strictly above a rigorous range bound for the reduced unpenalized objective. It then verifies energy equality on every original state after deterministic ancilla lifting and checks that the projected QUBO ground portfolio matches the original HUBO ground portfolio.

## 9. Claim boundary

The eight-asset production recommendation is generated and validated classically. The reduced QAOA and higher-moment VQE experiments are controlled quantum-compatible benchmarks. At the problem sizes used here, exact enumeration is deliberately retained as ground truth. No result is presented as evidence of quantum advantage, and the project is not investment advice.
