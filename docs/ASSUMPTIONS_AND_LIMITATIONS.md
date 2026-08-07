# Assumptions, limitations, and next steps

## Assumptions

The flagship model uses deterministic synthetic/anonymized asset-class assumptions rather than restricted Vanguard data. Expected returns, covariance, income yields, transaction-cost rates, the current portfolio, and stress scenarios are fixed in code so the result is reproducible.

The production exact-grid benchmark uses 10% allocation increments. The reduced QUBO uses a coarser 12.5% allocation unit and five representative sleeves. These are deliberate problem-size choices for an exactly auditable prototype, not claims about a production investment platform.

Transaction costs are reported using raw cost rates applied to gross absolute weight traded. The optimization uses a positive-semidefinite cost-tilted quadratic rebalancing proxy so the flagship objective remains quadratic and convex. One-way turnover is defined as half of gross absolute turnover.

The higher-moment budget-aligned model uses eight integer allocation units and a `$1,250` unit value for a `$10,000` encoded budget. The original `$1,150` source geometry is retained only as a diagnostic because it cannot represent `$10,000` exactly.

## Limitations

**Market realism.** Synthetic assumptions demonstrate methodology and reproducibility but do not constitute a forecast or a calibrated institutional capital-market model.

**Discretization.** The exact 10% production grid and 12.5% reduced quantum grid introduce quantization error. The continuous-vs-grid objective gaps reported in the audit quantify part of this effect.

**Quantum scale.** The reduced QUBO and 12/15-qubit HUBO instances are small enough for exact classical enumeration. The project therefore benchmarks quantum-compatible algorithms against known answers; it does not demonstrate quantum computational advantage.

**VQE/QAOA variability.** Variational methods depend on ansatz expressivity, optimizer budget, initialization, and finite-shot sampling. The checkpoint machinery makes this sensitivity visible rather than hiding it.

**Higher-moment estimation.** Co-skewness and especially co-kurtosis are sample-sensitive. The research appendix is therefore presented as an exploratory extension, not as the primary investment recommendation.

**Five-asset 15-qubit provenance.** A single fully observed joint five-asset tensor was not available for the stitched 15-qubit demonstration. Terms containing both TSLA and NVDA are unavailable in the source combination and are set to zero. The 12-qubit full-tensor case is the cleaner higher-moment benchmark.

**Quadratization overhead.** Exact HUBO-to-QUBO reduction trades higher-order interactions for ancilla variables and potentially large penalty scales. The dashboard reports this overhead explicitly.

## Recommended next steps

A production continuation would replace synthetic assumptions with approved institutional forecasts and cost models, expand the hard-constraint set, introduce rolling-time backtests, evaluate robustness under estimation error, and use decomposition/encoding techniques for larger universes. On the quantum side, useful next experiments include noise models or hardware runs, ansatz/optimizer benchmarking, sparse or structured quadratization, and comparisons against modern mixed-integer and specialized classical heuristics at increasing problem size.

## Claim boundary

The project is a research and demonstration prototype. The flagship allocation is classically solved and hard-constraint audited. Quantum algorithms are evaluated as separate reduced or exploratory benchmarks. No result is investment advice or evidence of quantum advantage.
