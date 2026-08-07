"""Checkpoint sweep for final VQE optimizer budgets on a fixed learned HUBO."""
from __future__ import annotations
from typing import Any, Sequence
from .higher_moment_extension import SourceMomentData
from .source_hubo_models import SourceHuboConfig, exact_source_hubo_report, run_source_hubo_vqe

def run_vqe_checkpoint_sweep(data:SourceMomentData,config:SourceHuboConfig,checkpoints:Sequence[int],*,shots:int=4096,seed:int=42,restarts:int=2)->dict[str,Any]:
    ordered=sorted({int(x) for x in checkpoints if int(x)>0})
    if not ordered: raise ValueError("At least one positive VQE optimizer checkpoint is required.")
    exact=exact_source_hubo_report(data,config); financial_index=int(exact["financial_reference"]["state_index"]); exact_index=int(exact["best_admissible_hamiltonian_state"]["state_index"]); financial_sharpe=float(exact["financial_reference"]["sharpe_ratio"]); rows=[]; reports={}
    for maxiter in ordered:
        report=run_source_hubo_vqe(data,config,maxiter=maxiter,shots=shots,seed=seed,restarts=restarts); reports[maxiter]=report; selected=report["selected"]; meta=report["metadata"]
        rows.append({"maxiter":int(maxiter),"bitstring":str(selected["bitstring"]),"state_index":int(selected["state_index"]),"runtime_seconds":float(report["runtime_seconds"]),"optimality_gap":float(meta["optimality_gap"]),"exact_hubo_ground_recovered":bool(meta["exact_admissible_optimum_recovered"]),"financial_ground_truth_recovered":bool(int(selected["state_index"])==financial_index),"feasible_probability_mass":float(meta["admissible_probability_mass"]),"exact_ground_probability":float(meta["exact_admissible_optimum_probability"]),"sharpe_ratio":float(selected["sharpe_ratio"]),"sharpe_abs_error":float(abs(float(selected["sharpe_ratio"])-financial_sharpe)),"classical_fallback_used":bool(meta["classical_fallback_used"])})
    recovered=[row for row in rows if row["exact_hubo_ground_recovered"] and not row["classical_fallback_used"]]
    if recovered:
        chosen=min(recovered,key=lambda row:(row["maxiter"],row["runtime_seconds"])); reason="smallest optimizer budget that recovered the exact feasible HUBO ground state"
    else:
        nonfallback=[row for row in rows if not row["classical_fallback_used"]] or rows; chosen=min(nonfallback,key=lambda row:(row["optimality_gap"],row["sharpe_abs_error"],row["maxiter"],row["runtime_seconds"])); reason="minimum exact Hamiltonian optimality gap (then Sharpe error and runtime)"
    selected_maxiter=int(chosen["maxiter"])
    return {"status":"completed","checkpoints":rows,"selected_maxiter":selected_maxiter,"selection_reason":reason,"selected_report":reports[selected_maxiter],"exact_feasible_ground_state_index":exact_index,"financial_ground_truth_state_index":financial_index,"shots":int(shots),"seed":int(seed),"restarts":int(restarts)}
