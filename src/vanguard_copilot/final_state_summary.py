"""Persist and summarize the latest automatic HUBO final state."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
DEFAULT_SUMMARY_PATH=Path("output/latest_hubo_selection.json")
def build_final_state_summary(*,qubits:int,sweep:Mapping[str,Any],exact_report:Mapping[str,Any],vqe_report:Mapping[str,Any]|None)->dict[str,Any]:
    financial=exact_report["financial_reference"]; exact_feasible=exact_report["best_admissible_hamiltonian_state"]; unrestricted=exact_report["hamiltonian_ground_state"]; final_state=vqe_report["selected"] if vqe_report is not None else exact_feasible; final_source="Finite-shot VQE selected state" if vqe_report is not None else "Exact feasible HUBO ground state"
    def _weights(record): return {str(asset):float(weight) for asset,weight in zip(exact_report["assets"],record["weights"])}
    summary={"model":f"Full-tensor budget-aligned HUBO — {int(qubits)} qubits","selected_generation":int(sweep["selected_generation"]),"selection_objective":str(sweep["objective_label"]),"selected_lambdas":{k:float(v) for k,v in sweep["selected_lambdas"].items()},"training_seconds":float(sweep["training_seconds"]),"selection_seconds":float(sweep["total_selection_seconds"]),"validation":{k:float(v) for k,v in sweep["selected_validation_metrics"].items()},"held_out":{k:float(v) for k,v in sweep["held_out_metrics"].items()},"financial_ground_truth":{"bitstring":str(financial["bitstring"]),"weights":_weights(financial),"expected_return":float(financial["expected_return"]),"volatility":float(financial["volatility"]),"sharpe_ratio":float(financial["sharpe_ratio"])},"exact_feasible_hubo_ground":{"bitstring":str(exact_feasible["bitstring"]),"weights":_weights(exact_feasible),"expected_return":float(exact_feasible["expected_return"]),"volatility":float(exact_feasible["volatility"]),"sharpe_ratio":float(exact_feasible["sharpe_ratio"]),"budget_breach":float(exact_feasible["budget_breach"])},"unrestricted_hamiltonian_ground":{"bitstring":str(unrestricted["bitstring"]),"budget_breach":float(unrestricted["budget_breach"]),"matches_financial_ground_truth":bool(unrestricted["state_index"]==financial["state_index"])},"final_state_source":final_source,"final_state":{"bitstring":str(final_state["bitstring"]),"weights":_weights(final_state),"expected_return":float(final_state["expected_return"]),"volatility":float(final_state["volatility"]),"sharpe_ratio":float(final_state["sharpe_ratio"]),"budget_breach":float(final_state.get("budget_breach",0.0)),"return_difference_vs_ground_truth":float(final_state["expected_return"]-financial["expected_return"]),"volatility_difference_vs_ground_truth":float(final_state["volatility"]-financial["volatility"]),"sharpe_difference_vs_ground_truth":float(final_state["sharpe_ratio"]-financial["sharpe_ratio"]),"matches_financial_ground_truth":bool(final_state["state_index"]==financial["state_index"]),"matches_exact_feasible_hubo_ground":bool(final_state["state_index"]==exact_feasible["state_index"])},"vqe":None}
    if vqe_report is not None:
        meta=vqe_report["metadata"]; summary["vqe"]={"status":str(vqe_report["status"]),"optimizer_evaluations":int(meta["maxiter"]),"shots":int(meta["shots"]),"runtime_seconds":float(vqe_report["runtime_seconds"]),"optimality_gap":float(meta["optimality_gap"]),"feasible_probability_mass":float(meta["admissible_probability_mass"]),"exact_hubo_ground_recovered":bool(meta["exact_admissible_optimum_recovered"]),"classical_fallback_used":bool(meta["classical_fallback_used"])}
    return summary
def save_final_state_summary(summary:Mapping[str,Any],path:Path|str=DEFAULT_SUMMARY_PATH)->Path:
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"); return target
def load_final_state_summary(path:Path|str=DEFAULT_SUMMARY_PATH)->dict[str,Any]|None:
    target=Path(path)
    if not target.exists(): return None
    try: payload=json.loads(target.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return None
    return payload if isinstance(payload,dict) else None
