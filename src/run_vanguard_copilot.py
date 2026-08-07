"""Command-line entry point for the audited portfolio co-pilot."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from vanguard_copilot.higher_moment_extension import extension_report
from vanguard_copilot.workflow import challenge_report,run_challenge

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--profile",choices=("Growth","Balanced","Defensive"),default="Balanced"); parser.add_argument("--output",default="output/vanguard_copilot_report.json"); parser.add_argument("--discrete-step",type=float,default=0.10); parser.add_argument("--quantum",action="store_true"); parser.add_argument("--qaoa-p",type=int,default=1); parser.add_argument("--qaoa-maxiter",type=int,default=60); parser.add_argument("--shots",type=int,default=4096); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--exploratory-higher-moment",action="store_true"); parser.add_argument("--higher-moment-vqe",action="store_true"); parser.add_argument("--higher-moment-maxiter",type=int,default=80); args=parser.parse_args()
    run=run_challenge(profile_name=args.profile,discrete_step=args.discrete_step,run_quantum=args.quantum,qaoa_p=args.qaoa_p,qaoa_maxiter=args.qaoa_maxiter,qaoa_shots=args.shots,seed=args.seed); report=challenge_report(run)
    if args.exploratory_higher_moment or args.higher_moment_vqe: report["exploratory_higher_moment_extension"]=extension_report(run_vqe=args.higher_moment_vqe,maxiter=args.higher_moment_maxiter,shots=args.shots,seed=args.seed)
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(report,indent=2),encoding="utf-8"); m=report["production_model"]["discrete_exact"]["metrics"]
    print("\nVANGUARD-ALIGNED MULTI-ASSET PORTFOLIO CO-PILOT"); print(f"Profile: {args.profile}"); print(f"Expected return: {100*m['expected_return']:.2f}%"); print(f"Volatility: {100*m['volatility']:.2f}%"); print(f"Risk-adjusted ratio: {m['sharpe_like']:.4f}"); print(f"Hard breaches: {len(m['hard_breaches'])}"); print(f"Report written to {destination}")
if __name__=="__main__": main()
