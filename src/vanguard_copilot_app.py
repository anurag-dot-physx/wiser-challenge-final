"""Streamlit prototype for the audited multi-asset portfolio co-pilot."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from vanguard_copilot.model import PROFILES
from vanguard_copilot.workflow import challenge_report,comparison_rows,run_challenge
st.set_page_config(page_title="Quantum Portfolio Co-Pilot",page_icon="⚛️",layout="wide")
st.title("Quantum-Enhanced Multi-Asset Portfolio Co-Pilot")
st.caption("WISER × Vanguard 2026 submission prototype. Synthetic/anonymized asset-class data; audited hard guardrails; reduced QUBO/QAOA benchmark reported separately from the eight-asset production model.")
with st.sidebar:
    st.header("Investor goals"); profile_name=st.selectbox("Profile",list(PROFILES),index=1); base=PROFILES[profile_name]
    return_weight=st.slider("Growth priority",0.5,5.0,float(base.return_weight),0.1); risk_weight=st.slider("Risk control",0.5,5.0,float(base.risk_weight),0.1); income_weight=st.slider("Income priority",0.0,2.0,float(base.income_weight),0.05); drawdown_weight=st.slider("Scenario / drawdown control",0.0,3.0,float(base.drawdown_weight),0.05); turnover_weight=st.slider("Cost-aware rebalancing sensitivity",0.0,2.0,float(base.turnover_weight),0.05)
    st.header("Guardrails"); equity_max=st.slider("Maximum equity",0.35,0.90,float(base.equity_max),0.05); defensive_min=st.slider("Minimum defensive allocation",0.10,0.60,float(base.defensive_min),0.05); alternatives_max=st.slider("Maximum alternatives",0.05,0.35,float(base.alternatives_max),0.05); asset_max=st.slider("Maximum single asset class",0.20,0.60,float(base.asset_max),0.05)
    st.header("Quantum demonstration"); run_quantum=st.checkbox("Run reduced exact-constraint QAOA simulation",value=False); qaoa_p=st.select_slider("QAOA depth",options=[1,2],value=1); qaoa_maxiter=st.slider("QAOA optimizer evaluations",20,120,50,10); qaoa_shots=st.select_slider("Finite shots",options=[1024,2048,4096,8192],value=4096); run_button=st.button("Optimize portfolio",type="primary",use_container_width=True)
if not run_button:
    st.info("Choose goals and guardrails, then select **Optimize portfolio**."); st.stop()
try:
    run=run_challenge(profile_name=profile_name,profile_overrides={"return_weight":return_weight,"risk_weight":risk_weight,"income_weight":income_weight,"drawdown_weight":drawdown_weight,"turnover_weight":turnover_weight,"equity_max":equity_max,"defensive_min":defensive_min,"alternatives_max":alternatives_max,"asset_max":asset_max},run_quantum=run_quantum,qaoa_p=qaoa_p,qaoa_maxiter=qaoa_maxiter,qaoa_shots=qaoa_shots)
except Exception as exc:
    st.error(f"Optimization failed: {type(exc).__name__}: {exc}"); st.stop()
report=challenge_report(run); production=report["production_model"]; discrete=production["discrete_exact"]; metrics=discrete["metrics"]; columns=st.columns(5); columns[0].metric("Expected return",f"{100*metrics['expected_return']:.2f}%"); columns[1].metric("Volatility",f"{100*metrics['volatility']:.2f}%"); columns[2].metric("Risk-adjusted ratio",f"{metrics['sharpe_like']:.3f}"); columns[3].metric("One-way turnover",f"{100*metrics['turnover']:.1f}%"); columns[4].metric("Hard breaches",str(len(metrics["hard_breaches"])))
st.subheader("Recommended production allocation"); allocation_rows=[]
for method_key,method_label in (("baseline","Current portfolio"),("continuous","Continuous convex QP"),("discrete_exact","Discrete exact grid")):
    for asset,weight in production[method_key]["weights"].items(): allocation_rows.append({"Method":method_label,"Asset class":asset,"Weight":weight})
allocation_df=pd.DataFrame(allocation_rows); figure=px.bar(allocation_df,x="Asset class",y="Weight",color="Method",barmode="group",text_auto=".0%"); figure.update_yaxes(tickformat=".0%"); st.plotly_chart(figure,use_container_width=True)
st.subheader("Trade-offs versus baseline"); comparison=pd.DataFrame(comparison_rows(run)); percent_columns=["Expected return","Volatility","Income yield","One-way turnover","Gross turnover","Estimated cost","Worst scenario loss"]; st.dataframe(comparison.style.format({**{column:"{:.2%}" for column in percent_columns},"Risk-adjusted ratio":"{:.3f}","Objective":"{:.6f}","Runtime (s)":"{:.4f}"}),use_container_width=True)
left,right=st.columns(2)
with left:
    st.subheader("Guardrail validation"); weights=discrete["weights"]; equity=sum(weights[name] for name in ("US Equity","International Equity","Emerging Markets")); defensive=sum(weights[name] for name in ("Government Bonds","Corporate Bonds","Cash")); alternatives=sum(weights[name] for name in ("Commodities","Real Estate")); guardrails=pd.DataFrame([{"Guardrail":"Fully invested","Actual":sum(weights.values()),"Limit":1.0,"Status":"PASS"},{"Guardrail":"Equity maximum","Actual":equity,"Limit":equity_max,"Status":"PASS" if equity<=equity_max+1e-9 else "FAIL"},{"Guardrail":"Defensive minimum","Actual":defensive,"Limit":defensive_min,"Status":"PASS" if defensive>=defensive_min-1e-9 else "FAIL"},{"Guardrail":"Alternatives maximum","Actual":alternatives,"Limit":alternatives_max,"Status":"PASS" if alternatives<=alternatives_max+1e-9 else "FAIL"},{"Guardrail":"Single asset maximum","Actual":max(weights.values()),"Limit":asset_max,"Status":"PASS" if max(weights.values())<=asset_max+1e-9 else "FAIL"}]); st.dataframe(guardrails.style.format({"Actual":"{:.1%}","Limit":"{:.1%}"}),hide_index=True,use_container_width=True)
with right:
    st.subheader("Why this allocation")
    for reason in production["explanation"]: st.write("•",reason)
st.divider(); st.subheader("Quantum-compatible reduced model"); reduced=report["reduced_quantum_model"]
if reduced["status"]!="available": st.warning("The production model is feasible, but the selected guardrails cannot be represented on the coarse reduced quantum grid. The production recommendation remains valid."); st.code(reduced["reason"] or "Reduced encoding unavailable")
else:
    audit=reduced["qubo"]["audit"]; qcols=st.columns(5); qcols[0].metric("Total qubits",str(reduced["n_qubits"])); qcols[1].metric("Allocation / slack",f"{reduced['allocation_qubits']} / {reduced['slack_qubits']}"); qcols[2].metric("Allocation step",f"{100*reduced['allocation_step']:.1f}%"); qcols[3].metric("Feasible portfolios",str(audit["unique_hard_feasible_portfolios"])); qcols[4].metric("QUBO ground audited",str(audit["ground_portfolio_matches_reduced_exact"])); reduced_rows=[{"Method":"Reduced exact","Asset class":name,"Weight":weight} for name,weight in reduced["exact_classical"]["weights"].items()]
    if "qaoa" in reduced: reduced_rows.extend({"Method":"QAOA / fallback","Asset class":name,"Weight":weight} for name,weight in reduced["qaoa"]["weights"].items())
    rdf=pd.DataFrame(reduced_rows); qfig=px.bar(rdf,x="Asset class",y="Weight",color="Method",barmode="group",text_auto=".1%"); qfig.update_yaxes(tickformat=".0%"); st.plotly_chart(qfig,use_container_width=True)
    if "qaoa" in reduced:
        meta=reduced["qaoa"]["metadata"]; st.write(f"QAOA status: **{reduced['qaoa']['status']}** · feasible mass: **{100*meta['feasible_probability_mass']:.2f}%** · exact reduced portfolio recovered: **{meta['exact_reduced_recovered']}** · classical fallback: **{meta['classical_fallback_used']}**")
with st.expander("Mathematical formulation"):
    st.latex(r"J(w)=\lambda_r w^T\Sigma w-\lambda_\mu\mu^T w-\lambda_y y^T w+\lambda_T(w-w_0)^TR(w-w_0)+\lambda_s\frac{1}{S}\sum_s(\ell_s^T w)^2"); st.latex(r"E(x,s)=J(Ax)+P\sum_k(c_k^Tx+d_k^Ts_k-t_k)^2")
