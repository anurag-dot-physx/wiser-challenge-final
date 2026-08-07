"""Higher-moment HUBO/VQE research appendix dashboard."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from vanguard_copilot.five_asset_hubo import load_five_asset_snapshot
from vanguard_copilot.higher_moment_extension import HigherMomentConfig, exact_higher_moment_benchmark, load_source_snapshot, run_higher_moment_vqe
from vanguard_copilot.source_hubo_models import SourceHuboConfig, exact_source_hubo_report, run_source_hubo_vqe

st.set_page_config(page_title="Higher-Moment HUBO/VQE",page_icon="🧪",layout="wide")
st.title("Exploratory Higher-Moment HUBO / VQE")
st.caption("Research appendix: co-skewness and co-kurtosis extend the mean-variance objective. This is separate from the flagship eight-asset production claim.")
with st.sidebar:
    qubits=st.radio("Model size",[12,15],horizontal=True); model=st.radio("HUBO formulation",["Normalized-moment original","Full-tensor budget-aligned"]); run_vqe=st.checkbox("Run finite-shot VQE",False); maxiter=st.slider("Optimizer evaluations",20,200,80,20); shots=st.select_slider("Shots",[1024,2048,4096,8192],4096); seed=st.number_input("Seed",0,1_000_000,42); run=st.button("Evaluate HUBO",type="primary",use_container_width=True)
if not run:st.info("Choose a model and evaluate its exact benchmark; enable VQE only when needed.");st.stop()
data=load_source_snapshot() if qubits==12 else load_five_asset_snapshot()
if model=="Normalized-moment original":
    cfg=HigherMomentConfig(); exact=exact_higher_moment_benchmark(data,cfg); st.success(f"Exact enumeration: {exact['states_enumerated']:,} states; {exact['feasible_states']:,} feasible."); records=[("Mean-variance optimum",exact["mean_variance_optimum"]),("Higher-moment optimum",exact["higher_moment_optimum"]),("Unrestricted HUBO ground",exact["hubo_ground_state"])]
    vqe=None
    if run_vqe:vqe=run_higher_moment_vqe(data,cfg,maxiter=int(maxiter),shots=int(shots),seed=int(seed));records.append(("Finite-shot VQE",{"bitstring":vqe.selected_bitstring,**vqe.selected_metrics,"weights":vqe.selected_weights.tolist(),"fully_invested":True}))
else:
    cfg=SourceHuboConfig(mode="budget_aligned"); exact=exact_source_hubo_report(data,cfg); st.success(f"Exact enumeration: {exact['states_enumerated']:,} states; exact $10,000 budget representable={exact['exact_budget_representable']}."); records=[("Financial ground truth",exact["financial_reference"]),("Exact feasible HUBO ground",exact["best_admissible_hamiltonian_state"]),("Unrestricted Hamiltonian ground",exact["hamiltonian_ground_state"])]
    vqe=None
    if run_vqe:vqe=run_source_hubo_vqe(data,cfg,maxiter=int(maxiter),shots=int(shots),seed=int(seed));records.append(("Finite-shot VQE",vqe["selected"]))
rows=[]
for label,r in records:rows.append({"State":label,"Bitstring":r["bitstring"],"Expected return":r["expected_return"],"Volatility":r["volatility"],"Sharpe":r["sharpe_ratio"],"Feasible / minimum breach":r.get("fully_invested",r.get("minimum_breach_state",False)),"Budget breach":r.get("budget_breach",0.0)})
st.dataframe(pd.DataFrame(rows).style.format({"Expected return":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.5f}","Budget breach":"${:,.2f}"}),hide_index=True,use_container_width=True)
st.subheader("Allocation weights")
alloc=[]
for label,r in records:
    for asset,weight in zip(data.tickers,r["weights"]):alloc.append({"State":label,"Asset":asset,"Weight":weight})
st.dataframe(pd.DataFrame(alloc).style.format({"Weight":"{:.1%}"}),hide_index=True,use_container_width=True)
if vqe is not None:
    metadata=vqe.metadata if hasattr(vqe,"metadata") else vqe["metadata"]; st.subheader("VQE diagnostics");st.json(dict(metadata))
st.caption("No quantum-advantage claim is made. Exact enumeration provides the benchmark at these problem sizes.")
