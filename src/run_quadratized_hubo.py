"""Build/export an exact quadratized QUBO for the full-tensor portfolio HUBO."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import numpy as np
from vanguard_copilot.final_state_summary import load_final_state_summary
from vanguard_copilot.five_asset_hubo import load_five_asset_snapshot
from vanguard_copilot.higher_moment_extension import load_source_snapshot
from vanguard_copilot.hubo_quadratization import compare_qubo_with_hubo,export_qubo_upper_triangle,quadratize_portfolio_hubo
from vanguard_copilot.source_hubo_models import SourceHuboConfig

def _json_safe(value:Any)->Any:
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value
def main():
    p=argparse.ArgumentParser();p.add_argument("--qubits",type=int,choices=(12,15),default=12);p.add_argument("--max-ancillas",type=int,default=None);p.add_argument("--penalty-safety-factor",type=float,default=1.05);p.add_argument("--full-enumeration-limit",type=int,default=22);p.add_argument("--output-dir",type=Path,default=Path("output/quadratized_hubo"));p.add_argument("--use-latest-learned-lambdas",action="store_true");args=p.parse_args(); data=load_source_snapshot() if args.qubits==12 else load_five_asset_snapshot();config=SourceHuboConfig(mode="budget_aligned")
    if args.use_latest_learned_lambdas:
        summary=load_final_state_summary()
        if summary and f"{args.qubits} qubits" in str(summary.get("model","")):
            l=summary.get("selected_lambdas",{});config=SourceHuboConfig(mode="budget_aligned",lambda_return=float(l["return"]),lambda_variance=float(l["variance"]),lambda_skewness=float(l["skewness"]),lambda_kurtosis=float(l["kurtosis"]),lambda_budget=float(l["budget"]))
    quadratized=quadratize_portfolio_hubo(data,config,max_ancillas=args.max_ancillas,penalty_safety_factor=args.penalty_safety_factor);audit=compare_qubo_with_hubo(data,config,quadratized,max_full_enumeration_variables=args.full_enumeration_limit);args.output_dir.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output_dir/f"hubo_qubo_{args.qubits}q.npz",Q=quadratized.Q,offset=np.asarray(quadratized.offset));upper=export_qubo_upper_triangle(quadratized);metadata={"model":f"budget-aligned full-tensor HUBO {args.qubits}q","lambdas":{"return":config.lambda_return,"variance":config.lambda_variance,"skewness":config.lambda_skewness,"kurtosis":config.lambda_kurtosis,"budget":config.lambda_budget},"n_original_variables":quadratized.n_original_variables,"n_ancillas":quadratized.n_ancillas,"n_total_variables":quadratized.n_total_variables,"penalty_strength":quadratized.penalty_strength,"objective_range_bound":quadratized.objective_range_bound,"constraints":[{"ancilla":c.ancilla,"left":c.left,"right":c.right} for c in quadratized.constraints],"qubo_upper_triangle":{f"{i},{j}":coefficient for (i,j),coefficient in upper.items()},"offset":quadratized.offset,"audit":audit};(args.output_dir/f"hubo_qubo_{args.qubits}q.json").write_text(json.dumps(_json_safe(metadata),indent=2,sort_keys=True),encoding="utf-8");print(json.dumps({k:metadata[k] for k in ("n_original_variables","n_ancillas","n_total_variables","penalty_strength","objective_range_bound")},indent=2));print("Energy equivalent:",audit["energy_equivalence_passed"],"ground match:",audit["lifted_ground_matches_hubo"])
if __name__=="__main__":main()
