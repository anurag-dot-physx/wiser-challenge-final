"""Run the source-derived higher-moment VQE exploratory extension."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from vanguard_copilot.higher_moment_extension import extension_report
def main():
    p=argparse.ArgumentParser();p.add_argument("--vqe",action="store_true");p.add_argument("--maxiter",type=int,default=80);p.add_argument("--shots",type=int,default=4096);p.add_argument("--seed",type=int,default=42);p.add_argument("--output",default="output/exploratory_higher_moment_extension.json");args=p.parse_args();report=extension_report(run_vqe=args.vqe,maxiter=args.maxiter,shots=args.shots,seed=args.seed);destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2),encoding="utf-8");exact=report["exact_benchmark"];print(f"Assets: {', '.join(exact['assets'])} | qubits: {exact['n_qubits']} | feasible states: {exact['feasible_states']}");print(f"Mean-variance optimum: {exact['mean_variance_optimum']['bitstring']}");print(f"Higher-moment optimum: {exact['higher_moment_optimum']['bitstring']}");print(f"Report written to {destination}")
if __name__=="__main__":main()
