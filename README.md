# Quantum-Enhanced Multi-Asset Portfolio Co-Pilot

**WISER x Vanguard - Quantum for Finance Challenge 2026**

[![Portfolio co-pilot tests](https://github.com/anurag-dot-physx/wiser-challenge-final/actions/workflows/tests.yml/badge.svg)](https://github.com/anurag-dot-physx/wiser-challenge-final/actions/workflows/tests.yml)

This project explores how classical portfolio optimization and quantum-compatible optimization can work together in a practical multi-asset portfolio construction workflow. Our goal was not simply to build a quantum model in isolation, but to develop a complete pipeline that starts from an interpretable portfolio problem, validates every important result classically, and then extends the same problem toward richer higher-order objectives and variational quantum optimization.

> **Main result:** for the flagship eight-asset model, the complete 10% allocation grid contains **19,448** fully invested portfolios. Across the three canonical investor profiles, the continuous problem is convex, the exact-grid recommendation satisfies all hard investment guardrails, and the reduced QUBO benchmark reproduces its own exact reduced classical optimum.

The quantum experiments are therefore reported against well-defined classical references. We do **not** claim quantum advantage, and this project is not investment advice.

## Presentation

- [Presentation deck - PDF](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pdf)
- [Presentation deck - PowerPoint](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pptx)

## 1. The problem we set out to solve

The challenge asks for a portfolio recommendation that balances expected return, risk, income, implementation cost and drawdown control while remaining inside clear investment guardrails. We approached this as both a portfolio-construction problem and a model-design problem: the optimizer should be useful and explainable on its own, but it should also admit quantum-compatible formulations that can be benchmarked rigorously.

Our overall workflow is:

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
 higher-moment HUBO + classical learning + VQE
```

This lets us keep the business-facing recommendation, the exact classical checks, and the quantum experiments connected without confusing one benchmark for another.

## 2. Why we selected this approach

We selected a hybrid architecture because it lets different parts of the problem contribute where they are strongest.

The **quadratic portfolio model** gives us an interpretable and efficiently auditable foundation. It captures the main portfolio trade-offs required by the challenge, keeps the production problem convex, supports explicit hard guardrails, and allows us to compare the continuous solution with a complete discrete enumeration. This makes it a strong baseline for both financial interpretation and technical validation.

The **reduced QUBO formulation** then gives us a clean quantum-compatible benchmark. By reducing the representative universe to a size that can still be enumerated exactly, we know the true optimum before running QAOA. That makes the quantum result meaningful: we can measure optimality gap, feasible probability mass and recovery of the exact ground portfolio rather than judging the result only qualitatively.

The **higher-moment HUBO extension** was chosen because co-skewness and co-kurtosis introduce structure that a quadratic mean-variance model cannot represent. Although these higher-order interactions increase hardware and optimization complexity, they also enrich the energy landscape substantially. This richer landscape may capture asymmetry, tail behaviour and cross-asset effects that are invisible to a purely quadratic objective.

A second motivation for the higher-moment route is the hybrid classical-quantum philosophy of the program itself. In our workflow, the Hamiltonian coefficients are first optimized classically using **CMA-ES**, using training and validation data to select an effective set of coefficients. Once the Hamiltonian is fixed, **VQE** is used for the quantum optimization stage. This synchronization of classical learning and quantum optimization was one of the main reasons we selected this specific approach for incorporating higher-moment corrections.

The project therefore combines three useful strengths in one workflow: an interpretable production model, exact classical references for validation, and a richer higher-order extension in which classical learning and quantum optimization work together rather than compete.

## 3. Flagship mathematical model

For portfolio weights $w$, the production objective is

$$
J(w)=
-\lambda_\mu\mu^T w
+\lambda_r w^T\Sigma w
-\lambda_y y^T w
+\lambda_T(w-w_0)^T R(w-w_0)
+\lambda_s\frac{1}{S}\sum_s(\ell_s^T w)^2.
$$

The terms reward expected return and income while penalizing variance, costly rebalancing and adverse scenario exposure. The matrix $R$ is positive semidefinite, and the scenario term is also quadratic, so the resulting production problem remains convex for the canonical profiles.

Hard guardrails enforce:

- full investment and long-only weights;
- a maximum single-asset allocation;
- a maximum aggregate equity exposure;
- a minimum defensive allocation;
- a maximum alternatives allocation.

The dashboard exposes **Growth, Balanced and Defensive** profiles, along with tunable growth, risk, income, drawdown and implementation-cost preferences.

## 4. Classical validation and verified results

We did not rely on the continuous optimizer alone. The 10% discrete benchmark exhaustively evaluates

$$
\binom{17}{7}=19{,}448
$$

fully invested portfolios, which gives us a complete reference for the production grid.

| Profile | Max form error | Min eig(H) | QP -> grid objective gap | Grid states | Reduced QUBO vars | Feasible reduced portfolios |
|---|---:|---:|---:|---:|---:|---:|
| Growth | 5.551e-17 | 1.010e-02 | 0.00081799 | 19,448 | 13 | 108 |
| Balanced | 6.939e-17 | 2.185e-02 | 0.00066947 | 19,448 | 15 | 66 |
| Defensive | 4.857e-17 | 3.365e-02 | 0.00140581 | 19,448 | 11 | 2 |

The audit also provides a useful economic sanity check: increasing the rebalancing-cost preference changes one-way turnover from **40% to 15%**, showing that the control has a material effect on the recommended portfolio.

The canonical verified summary is stored in [`results/flagship/quadratic_model_audit_summary.json`](results/flagship/quadratic_model_audit_summary.json).

To regenerate the audit:

```bash
python src/audit_quadratic_model.py
```

## 5. Reduced exact-constraint QUBO and QAOA

For the quantum-compatible benchmark, we encode a five-sleeve representative universe on a 12.5% allocation grid. Budget and group inequalities are converted into exact quadratic equalities using binary slack variables:

$$
E(x,s)=J(Ax)+P\sum_k\left(c_k^T x+d_k^T s_k-t_k\right)^2.
$$

The penalty $P$ is chosen above the full financial-energy range, so violating the encoded constraints cannot improve the global optimum. Because this reduced state space is still exactly enumerable, we can verify that the QUBO ground portfolio is hard-feasible and matches the exact reduced financial optimum before running QAOA.

This gives the quantum benchmark a particularly useful feature: every variational result can be compared with a known exact answer.

## 6. Portfolio co-pilot

The Streamlit dashboard is the business-facing layer of the project. It translates the optimization into quantities that are easier to interpret: allocation changes, expected return, volatility, risk-adjusted performance, income, turnover, cost, scenario loss and hard-constraint checks.

Launch it with:

```bash
streamlit run src/vanguard_copilot_app.py
```

The additional dashboard pages expose the higher-moment model, automatic generation selection, VQE checkpoint selection, final-state summaries and exact HUBO-to-QUBO quadratization.

## 7. Higher-moment HUBO/VQE extension

The higher-moment model adds complete co-skewness and co-kurtosis tensors:

$$
E(m)=
-\lambda_\mu\mu^T m
+\lambda_\Sigma m^T\Sigma m
-\lambda_S\sum_{ijk}S_{ijk}m_i m_j m_k
+\lambda_K\sum_{ijkl}K_{ijkl}m_i m_j m_k m_l
+\lambda_B\left(\sum_i m_i-8\right)^2.
$$

Three bits per asset encode integer allocation units. In the budget-aligned model, one unit is **USD 1,250**, so eight units represent the **USD 10,000** target exactly.

The main strength of this extension is expressiveness. Co-skewness and co-kurtosis allow the Hamiltonian to respond to asymmetry, tail structure and higher-order interactions between assets, producing a richer optimization landscape than the quadratic model alone.

The five Hamiltonian coefficients are calibrated with a vectorized **CMA-ES** outer loop. State features are cached once, so entire CMA populations can be evaluated efficiently with matrix multiplication. VQE is deliberately applied only after the classical calibration stage has fixed the Hamiltonian. This creates a clean classical-learning -> quantum-optimization pipeline and makes it possible to distinguish coefficient-calibration quality from quantum-solver quality.

### Training-depth selection

A single CMA trajectory is snapshotted at generation checkpoints such as

```text
5, 10, 20, 30, 50, 75, 100
```

Five sector universes are used for validation and model selection, while another five are held out until $G^\star$ and the Hamiltonian coefficients are fixed. The dashboard can select the checkpoint using highest Sharpe, highest return, lowest volatility, or the fastest near-optimal Sharpe criterion.

### Final VQE selection

After the Hamiltonian is fixed, VQE optimizer budgets can be swept over

```text
20, 40, 60, 80, 120, 160, 200
```

The preferred checkpoint is the smallest non-fallback run that recovers the exact feasible HUBO ground state; otherwise the selection is based on the exact Hamiltonian optimality gap and financial error metrics.

Generate the optional sector data with:

```bash
python src/fetch_sector_data.py
```

## 8. Exact HUBO to QUBO quadratization

The cubic and quartic pseudo-Boolean Hamiltonian can also be reduced exactly to a QUBO using reusable product ancillas. For each identity $y=ab$, we use the Rosenberg penalty

$$
P\left(ab-2ay-2by+3y\right).
$$

The penalty is chosen above a rigorous bound on the reduced unpenalized objective. The implementation then checks the stronger state-by-state identity

$$
H_{\mathrm{HUBO}}(x)=H_{\mathrm{QUBO}}\left(x,y(x)\right)
$$

for every original encoded state.

This gives us a useful bridge between the expressive higher-order model and standard quadratic binary hardware formulations: the higher-order financial structure can be retained while still producing an exact QUBO representation, at the price of ancillary variables.

Run the standalone audit/exporter with:

```bash
python src/run_quadratized_hubo.py
```

or use the latest learned Hamiltonian:

```bash
python src/run_quadratized_hubo.py --use-latest-learned-lambdas
```

## 9. Reproducing the project

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

A quick classical report is:

```bash
python src/run_vanguard_copilot.py --profile Balanced
```

A reduced QAOA demonstration is:

```bash
python src/run_vanguard_copilot.py --profile Balanced --quantum --qaoa-p 1 --qaoa-maxiter 60 --shots 4096 --seed 42
```

The GitHub Actions workflow performs the same installation from a fresh checkout, compiles the source, runs the regression suite and executes non-interactive smoke checks for the flagship audit and Balanced portfolio report.

## 10. Team members and contributions

**Anurag Sarkar** and **Ankit Gill** are the two team members who developed this project.

Both team members made **equal or comparable contributions** across the major stages of the work, including the development of the theoretical ideas, formulation of the portfolio objectives and constraints, design of the higher-order extensions, implementation of the classical and quantum optimization workflows, validation strategy, interpretation of results, and overall project development.

The project evolved through repeated discussion, testing and refinement, and both team members were thoroughly involved throughout that process. The ideas and implementations required sustained joint effort, so the work is best represented as a collaborative contribution rather than as a collection of isolated individual tasks.

## 11. AI and tools usage

The **working theories, higher-order extensions and core/bare implementations were developed by the team members**.

AI tools were used as supporting development tools, particularly for **code optimization, code organization and deployment tasks**, as well as for **preparing and formatting the presentation files**. The underlying modeling choices, mathematical formulations, higher-moment extensions and central implementation logic were developed and directed by the team.

This distinction is important to the project: AI-assisted tooling helped accelerate engineering and presentation work, while the scientific and methodological content remained driven by the team members.

## 12. Assumptions and limitations

The flagship uses deterministic synthetic/anonymized assumptions defined directly in code and does not rely on restricted Vanguard data. Raw implementation-cost rates are reported separately from the quadratic rebalancing proxy. Quantum finite-shot post-processing only selects among observed hard-feasible portfolios, and any classical fallback is reported explicitly.

The 15-qubit five-asset higher-moment example is a stitched scaling demonstration: mixed terms involving both TSLA and NVDA were unavailable in the supplied source combination and are therefore set to zero. The 12-qubit full-tensor case is the cleaner higher-moment benchmark.

See [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](docs/ASSUMPTIONS_AND_LIMITATIONS.md) for the full claim boundary and recommended next steps.

## 13. Repository guide

```text
src/                         application, solvers and quantum models
src/pages/                   Streamlit research and diagnostic pages
tests/                       submission-focused regression tests
data/                        data provenance and source snapshot
results/flagship/            canonical verified audit summary
docs/METHODOLOGY.md          detailed mathematical and algorithmic workflow
docs/RESULTS.md              verified findings
docs/ASSUMPTIONS_AND_LIMITATIONS.md
presentation/                PDF and PowerPoint submission deck
```

## 14. References and related work

The project draws on ideas from modern portfolio theory, binary optimization, variational quantum algorithms and higher-order portfolio modeling. The following references were particularly relevant to the formulation and methods used here:

1. **H. Markowitz, “Portfolio Selection,” The Journal of Finance 7, 77–91 (1952).** The classical mean–variance foundation underlying the quadratic portfolio model. [DOI: 10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)

2. **G. Kochenberger, J.-K. Hao, F. Glover, M. Lewis, Z. Lü, H. Wang, and Y. Wang, “The unconstrained binary quadratic programming problem: a survey,” Journal of Combinatorial Optimization 28, 58–81 (2014).** A broad reference for QUBO/UBQP modeling and solution methods. [Springer](https://link.springer.com/article/10.1007/s10878-014-9734-0)

3. **V. Uotila, J. Ripatti, and B. Zhao, “Higher-Order Portfolio Optimization with Quantum Approximate Optimization Algorithm,” 2025 IEEE International Conference on Quantum Computing and Engineering (QCE), pp. 1–12 (2025).** Especially relevant to our higher-moment extension: the work formulates portfolio optimization with skewness and kurtosis as a higher-order unconstrained binary optimization problem. [IEEE Xplore](https://ieeexplore.ieee.org/document/11249852/) · [DOI: 10.1109/QCE65121.2025.00244](https://doi.org/10.1109/QCE65121.2025.00244)

4. **E. Farhi, J. Goldstone, and S. Gutmann, “A Quantum Approximate Optimization Algorithm” (2014).** The foundational QAOA proposal used as the conceptual basis for our reduced QUBO quantum benchmark. [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)

5. **A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O’Brien, “A variational eigenvalue solver on a photonic quantum processor,” Nature Communications 5, 4213 (2014).** The foundational VQE paper underlying the variational solver used in the higher-moment workflow. [DOI: 10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213)

6. **A. Auger and N. Hansen, “CMA-ES: Evolution Strategies and Covariance Matrix Adaptation,” GECCO 2011 Companion, pp. 991–1010 (2011).** A useful methodological reference for the CMA-ES calibration stage used to learn the higher-moment Hamiltonian coefficients. [DOI: 10.1145/2001858.2002123](https://doi.org/10.1145/2001858.2002123)

These references are intended to situate the project within the relevant literature rather than imply that the implementation reproduces any one paper directly. Our contribution is the integrated workflow connecting an auditable constrained portfolio model, exact binary benchmarks, higher-moment Hamiltonians, classical coefficient learning, variational quantum optimization, and exact HUBO-to-QUBO reduction.

## 15. Key documents

- [Presentation deck - PDF](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pdf)
- [Presentation deck - PowerPoint](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pptx)
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](docs/ASSUMPTIONS_AND_LIMITATIONS.md)
- [`data/README.md`](data/README.md)
- [`results/flagship/quadratic_model_audit_summary.json`](results/flagship/quadratic_model_audit_summary.json)

---

**Claim boundary:** the eight-asset production recommendation is solved and audited classically for hard-constraint compliance. QAOA and VQE are evaluated on separately defined exactly enumerable benchmarks. No result establishes quantum advantage or constitutes investment advice.
