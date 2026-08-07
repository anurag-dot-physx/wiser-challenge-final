from pathlib import Path

path = Path("README.md")
text = path.read_text()
start = "<!-- RESULT_VISUALS_START -->"
end = "<!-- RESULT_VISUALS_END -->"
block = """<!-- RESULT_VISUALS_START -->

## Results at a glance

The three figures below summarize the strongest quantitative evidence in the submission. They deliberately separate the **production audit**, the **reduced QUBO/QAOA benchmark**, and the **exploratory higher-moment HUBO/VQE experiment** so that each result is compared only with its valid reference.

### 1. Flagship classical audit and turnover control

![Flagship audit results](results/figures/flagship_audit_results.svg)

The complete 19,448-state grid audit verifies zero canonical hard-constraint breaches. The small QP-to-grid objective gaps are consistent with discretizing a convex relaxation, while increasing rebalancing sensitivity reduces one-way turnover from **40% to 15%**.

### 2. Reduced exact-constraint QUBO/QAOA benchmark

![Reduced QUBO/QAOA allocation](results/figures/reduced_qaoa_allocation.svg)

For the Balanced reduced model, the audited QUBO ground portfolio matches the exact reduced optimum. The plotted allocation makes the recovery visually explicit while keeping this benchmark separate from the eight-asset production model.

### 3. Higher-moment HUBO/VQE research extension

![Higher-moment HUBO/VQE results](results/figures/higher_moment_vqe_results.svg)

The higher-moment experiment includes co-skewness and co-kurtosis. In the displayed 15-qubit exploratory run, the finite-shot VQE state remains budget-feasible but does **not** recover the exact HUBO ground state. This gap is reported directly as solver-quality evidence rather than as a quantum-advantage claim.

<!-- RESULT_VISUALS_END -->"""

if start in text and end in text:
    a = text.index(start)
    b = text.index(end) + len(end)
    text = text[:a] + block + text[b:]
else:
    anchor = "- [Presentation deck - PowerPoint](presentation/WISER_Vanguard_Quantum_Portfolio_Challenge_2026.pptx)"
    if anchor not in text:
        raise SystemExit("README presentation anchor not found")
    pos = text.index(anchor) + len(anchor)
    text = text[:pos] + "\n\n" + block + text[pos:]

path.write_text(text)
print("README result visuals updated")
