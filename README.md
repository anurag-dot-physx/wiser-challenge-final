# Quantum-Enhanced Multi-Asset Portfolio Co-Pilot

**WISER x Vanguard - Quantum for Finance Challenge 2026**

[![Portfolio co-pilot tests](https://github.com/anurag-dot-physx/wiser-challenge-final/actions/workflows/tests.yml/badge.svg)](https://github.com/anurag-dot-physx/wiser-challenge-final/actions/workflows/tests.yml)

This repository contains a submission-oriented prototype for **Multi-Asset Portfolio Construction**: an audited classical portfolio optimizer, an exact reduced QUBO/QAOA benchmark, and a higher-moment HUBO/VQE research extension.

> **Primary result:** the flagship eight-asset model is solved classically and validated against all original hard guardrails. The complete 10% grid of **19,448** fully invested portfolios is enumerated exactly. For all three canonical investor profiles, the continuous objective is convex, the exact-grid solution has **zero hard-constraint breaches**, and the reduced QUBO ground portfolio matches its own exact reduced classical optimum.

The project is designed as a genuinely hybrid classical-quantum workflow: classical optimization provides strong, interpretable references and learns effective Hamiltonian parameters, while quantum-compatible formulations explore richer discrete optimization landscapes. **No quantum-advantage claim is made, and this project is not investment advice.**

## Presentation

- [Presentation deck - PDF](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pdf)
- [Presentation deck - PowerPoint](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pptx)

## 1. Challenge and solution

The challenge asks for an interpretable quantum-compatible portfolio optimizer that balances expected return, risk, implementation cost, income and drawdown control while satisfying investment guardrails.

```text
synthetic/anonymized assumptions + investor goals
                |
                v
      convex quadratic portfolio model
                |
                v
 continuous QP + exact 10% grid validation
                |
                v
       recommended allocation + rationale
                |
                v
  reduced exact-constraint QUBO -> QAOA benchmark
                |
                v
 higher-moment HUBO + classical learning -> VQE
                |
                v
       exact HUBO -> QUBO quadratization
```

The project answers two complementary questions:

1. **Can the investment problem be formulated, solved and audited cleanly?** - addressed by the eight-asset quadratic production model.
2. **Can the portfolio objective be enriched beyond quadratic mean-variance structure while combining classical learning with quantum optimization?** - addressed by the reduced QUBO/QAOA and higher-moment HUBO/VQE workflows.

### Why we selected this approach

We selected this layered hybrid architecture because each part contributes a distinct strength to the overall portfolio-construction problem.

The **quadratic portfolio model** gives us a transparent and highly auditable foundation. Expected return, variance risk, income, implementation cost and scenario sensitivity all appear explicitly in the objective, while hard investment guardrails remain easy to interpret and verify. Convexity provides a strong continuous reference, and complete enumeration of the 10% allocation grid gives an exact discrete benchmark. This makes the flagship model useful not only as an optimizer, but also as a reliable validation layer for everything built on top of it.

The **reduced exact-constraint QUBO** then translates the same optimization philosophy into a native binary form. The reduced universe is small enough that its complete state space can be checked exactly, allowing QAOA to be compared with a known optimum rather than an unknown reference. This creates a controlled setting in which the behavior of a quantum optimizer can be studied without giving up financial interpretability or constraint auditing.

We were especially motivated to include **higher-moment corrections** because co-skewness and co-kurtosis introduce interactions that do not exist in a purely quadratic mean-variance objective. Although higher-order models naturally increase hardware and optimization complexity, they also **substantially enrich the energy landscape** of the portfolio problem. This richer landscape can represent asymmetric and higher-order cross-asset structure that a quadratic model cannot express directly, and therefore provides a natural setting in which quantum optimization may become particularly interesting.

A second major motivation is the **synchronization of classical learning and quantum optimization**. In one of our main higher-moment workflows, the Hamiltonian coefficients are first calibrated classically using a vectorized CMA-ES optimization over training and validation data. Once those coefficients are fixed, the resulting Hamiltonian is passed to VQE for quantum optimization. The classical stage therefore learns an effective objective, while the quantum stage focuses on exploring the resulting combinatorial energy landscape. This classical-to-quantum handoff closely reflects the hybrid philosophy that motivated the WISER program and was one of the main reasons we selected this specific architecture for the higher-moment extension.

The project also keeps **exact references wherever they are computationally available**. We calculate exact financial ground truths, exact feasible HUBO ground states and exact lifted HUBO-to-QUBO energy equivalence. This allows us to separate model quality, Hamiltonian calibration and quantum-solver performance instead of mixing them into a single metric. As a result, the project combines expressiveness with unusually strong internal validation.

Finally, the exact **HUBO-to-QUBO quadratization** provides a bridge between the richer higher-order model and standard quadratic binary hardware formulations. It shows that the added higher-order structure does not have to remain isolated from QUBO-based quantum methods: it can be reduced exactly using ancilla variables and rigorously chosen penalties while preserving the relevant ground-state structure.

Taken together, the final approach lets us move progressively from an interpretable classical portfolio model, to an exactly auditable binary quantum benchmark, to a richer higher-moment Hamiltonian in which classical parameter learning and quantum state optimization work together. That progression is central to the project: rather than treating classical and quantum optimization as competing alternatives, we use them as complementary components of one portfolio-construction workflow.

## 2. Flagship mathematical model

For portfolio weights **w**, the production objective is

$$
J(w)=
-\lambda_\mu\mu^T w
+\lambda_r w^T\Sigma w
-\lambda_y y^T w
+\lambda_T(w-w_0)^T R(w-w_0)
+\lambda_s\frac{1}{S}\sum_s(\ell_s^T w)^2.
$$

The five terms represent expected-return reward, variance risk, income reward, cost-aware rebalancing, and scenario/drawdown control. **R** is positive semidefinite, so the total Hessian remains positive semidefinite. The implementation separately reports raw transaction-cost estimates and one-way turnover.

Hard guardrails enforce:

- 100% investment and long-only weights;
- maximum single-asset weight;
- maximum aggregate equity exposure;
- minimum defensive exposure;
- maximum alternatives exposure.

The dashboard exposes **Growth, Balanced, and Defensive** profiles plus tunable growth, risk, income, drawdown, cost sensitivity and guardrails.

## 3. Classical validation and verified results

The continuous problem is solved as a constrained convex quadratic program. The production discrete benchmark uses 10% increments and exhaustively evaluates

$$
\binom{17}{7}=19{,}448
$$

fully invested portfolios.

| Profile | Max form error | Min eig(H) | QP -> grid objective gap | Grid states | Reduced QUBO vars | Feasible reduced portfolios |
|---|---:|---:|---:|---:|---:|---:|
| Growth | 5.551e-17 | 1.010e-02 | 0.00081799 | 19,448 | 13 | 108 |
| Balanced | 6.939e-17 | 2.185e-02 | 0.00066947 | 19,448 | 15 | 66 |
| Defensive | 4.857e-17 | 3.365e-02 | 0.00140581 | 19,448 | 11 | 2 |

The audit also changes one-way turnover from **40% to 15%** when the rebalancing preference is increased from zero to two, confirming that the cost-sensitivity control has a material effect.

Canonical verified summary: [`results/flagship/quadratic_model_audit_summary.json`](results/flagship/quadratic_model_audit_summary.json).

Regenerate the full audit with:

```bash
python src/audit_quadratic_model.py
```

## 4. Reduced exact-constraint QUBO

A five-sleeve representative universe is encoded on a 12.5% grid with bounded binary allocation variables. Budget and nonredundant group inequalities are converted into exact quadratic equalities using binary slack variables:

$$
E(x,s)=J(Ax)+P\sum_k\left(c_k^T x+d_k^T s_k-t_k\right)^2.
$$

The penalty **P** is automatically chosen above the complete financial-energy range. Exhaustive auditing verifies that the constraint-consistent QUBO state space corresponds to hard-feasible reduced portfolios and that every QUBO ground portfolio matches the exact reduced financial optimum.

QAOA is compared **only with this reduced exact reference**, not with the eight-asset production solution.

## 5. Portfolio co-pilot

Launch the interactive prototype:

```bash
streamlit run src/vanguard_copilot_app.py
```

The main page shows:

- recommended allocation versus the current portfolio and continuous reference;
- expected return, volatility, risk-adjusted ratio, income, turnover, cost and scenario loss;
- hard-guardrail validation;
- human-readable rationale;
- reduced QUBO/QAOA diagnostics.

Additional pages cover the higher-moment extension, automatic generation/VQE checkpoint selection, final-state summary, and exact HUBO-to-QUBO quadratization.

## 6. Higher-moment HUBO/VQE research extension

The higher-moment extension adds complete co-skewness and co-kurtosis tensors, giving the Hamiltonian access to asymmetric and higher-order cross-asset interactions beyond a quadratic mean-variance description:

$$
E(m)=
-\lambda_\mu\mu^T m
+\lambda_\Sigma m^T\Sigma m
-\lambda_S\sum_{ijk}S_{ijk}m_i m_j m_k
+\lambda_K\sum_{ijkl}K_{ijkl}m_i m_j m_k m_l
+\lambda_B\left(\sum_i m_i-8\right)^2.
$$

Three bits per asset encode integer allocation units. In the budget-aligned model, one unit is **USD 1,250**, so eight units exactly represent the **USD 10,000** target budget.

The five Hamiltonian coefficients are calibrated with a vectorized **CMA-ES** outer loop. State features are cached once and whole CMA populations are evaluated with matrix multiplication, making the classical learning stage efficient enough to explore many coefficient candidates. VQE is then applied only after the Hamiltonian has been selected, giving a clean separation between **learning the objective** and **quantum optimization of the objective**.

### Efficient training-depth selection

One CMA trajectory is snapshotted at generation checkpoints such as

```text
5, 10, 20, 30, 50, 75, 100
```

Five sector universes are used for validation/model selection; a separate five are held out until $G^\star$ and the coefficients are fixed. This creates a genuine model-selection pipeline rather than selecting hyperparameters on the final evaluation set. The dashboard can select highest Sharpe, highest return, lowest volatility, or the fastest near-optimal Sharpe checkpoint.

### Efficient final-VQE selection

After the Hamiltonian is fixed, VQE optimizer budgets can be swept over

```text
20, 40, 60, 80, 120, 160, 200
```

The preferred checkpoint is the smallest budget that recovers the exact feasible HUBO ground state without fallback; otherwise the lowest exact Hamiltonian optimality gap is used. This makes quantum runtime itself a tunable resource and lets us identify the smallest useful optimization budget instead of assuming that more iterations are always better.

Generate the optional sector data with:

```bash
python src/fetch_sector_data.py
```

## 7. Exact HUBO to QUBO quadratization

The full cubic/quartic pseudo-Boolean Hamiltonian can be reduced exactly to a QUBO with reusable product ancillas. For each identity $y=ab$, the Rosenberg penalty is

$$
P\left(ab-2ay-2by+3y\right).
$$

$P$ is selected strictly above a rigorous range bound for the reduced unpenalized objective. The implementation verifies the state-by-state condition

$$
H_{\mathrm{HUBO}}(x)=H_{\mathrm{QUBO}}\left(x,y(x)\right).
$$

for every original encoded state and confirms that the lifted QUBO ground portfolio matches the original HUBO ground portfolio. The quadratization therefore turns the richer higher-order objective into a standard QUBO while preserving exact energies on the consistent ancilla subspace and preserving the ground-state solution under the sufficient penalty construction.

Run the standalone audit/exporter with:

```bash
python src/run_quadratized_hubo.py
```

or quadratize the latest learned Hamiltonian after an automatic-selection run:

```bash
python src/run_quadratized_hubo.py --use-latest-learned-lambdas
```

## 8. Reproduce in a fresh environment

Python 3.11 is recommended.

```bash
git clone https://github.com/anurag-dot-physx/wiser-challenge-final.git
cd wiser-challenge-final
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python src/audit_quadratic_model.py
streamlit run src/vanguard_copilot_app.py
```

A quick non-quantum portfolio report is:

```bash
python src/run_vanguard_copilot.py --profile Balanced
```

A reduced QAOA demonstration is:

```bash
python src/run_vanguard_copilot.py --profile Balanced --quantum --qaoa-p 1 --qaoa-maxiter 60 --shots 4096 --seed 42
```

The GitHub Actions workflow performs the same installation from a fresh checkout, compiles the source, runs the regression suite, and executes non-interactive smoke checks for the flagship audit and Balanced portfolio report.

## 9. Repository guide

```text
src/                         application, solvers and quantum models
src/pages/                   Streamlit research/diagnostic pages
tests/                       submission-focused regression tests
data/                        synthetic/public-data provenance and snapshot
results/flagship/            canonical verified audit summary
docs/METHODOLOGY.md          detailed mathematical/algorithmic workflow
docs/RESULTS.md              verified findings
docs/ASSUMPTIONS_AND_LIMITATIONS.md
presentation/                PDF and PowerPoint submission deck
```

## 10. Assumptions, post-processing and limitations

The flagship uses deterministic synthetic/anonymized assumptions defined directly in code and does not rely on restricted Vanguard data. Raw implementation-cost rates are reported separately from the quadratic rebalancing proxy. Quantum finite-shot post-processing only selects among observed hard-feasible portfolios, and any classical fallback is reported.

The 15-qubit five-asset higher-moment example is a stitched scaling demonstration: mixed terms involving both TSLA and NVDA were unavailable in the supplied source combination and are set to zero. The 12-qubit full-tensor case is the cleaner higher-moment benchmark.

See [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](docs/ASSUMPTIONS_AND_LIMITATIONS.md) for the full claim boundary and recommended next steps.

## 11. Key documents

- [Presentation deck - PDF](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pdf)
- [Presentation deck - PowerPoint](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pptx)
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](docs/ASSUMPTIONS_AND_LIMITATIONS.md)
- [`data/README.md`](data/README.md)
- [`results/flagship/quadratic_model_audit_summary.json`](results/flagship/quadratic_model_audit_summary.json)

---

**Claim boundary:** the eight-asset production recommendation is classically solved and audited for hard-constraint compliance. QAOA and VQE are evaluated on separately defined exactly enumerable benchmarks. No result establishes quantum advantage or constitutes investment advice.