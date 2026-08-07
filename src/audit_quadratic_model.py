"""Run the mathematical and numerical audit of the flagship quadratic model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from vanguard_copilot.audit import quadratic_model_audit

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="output/quadratic_model_audit.json"); parser.add_argument("--samples",type=int,default=64); parser.add_argument("--seed",type=int,default=17); args=parser.parse_args(); report=quadratic_model_audit(random_seed=args.seed,samples=args.samples); destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n"+"="*118); print("FLAGSHIP QUADRATIC PORTFOLIO MODEL AUDIT".center(118)); print("="*118); print(f"Overall status         : {report['status'].upper()}"); print("-"*118); print(f"{'Profile':<12} {'form err':>12} {'min eig(H)':>12} {'QP-grid gap':>14} {'grid states':>12} {'QUBO qubits':>12} {'feasible portfolios':>20}")
    for name,row in report["profiles"].items(): print(f"{name:<12} {row['maximum_objective_form_error']:12.3e} {row['minimum_hessian_eigenvalue']:12.3e} {row['continuous_to_discrete_gap']:14.8f} {row['discrete_states']:12d} {row['reduced_qubits']:12d} {row['reduced_unique_feasible_portfolios']:20d}")
    sensitivity=report["tunable_goal_checks"]; print("-"*118); print(f"Turnover sensitivity  : {100*sensitivity['turnover_at_weight_0']:.1f}% -> {100*sensitivity['turnover_at_weight_2']:.1f}% (material={sensitivity['balanced_turnover_preference_material']})"); print("="*118); print(f"Report written to {destination}")
    if not report["all_checks_passed"]: raise SystemExit(1)
if __name__=="__main__": main()
