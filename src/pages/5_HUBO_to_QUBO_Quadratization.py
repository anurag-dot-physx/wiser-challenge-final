"""Dashboard for exact HUBO -> QUBO quadratization and ancilla overhead audits."""
from __future__ import annotations
import numpy as np,pandas as pd,plotly.express as px,streamlit as st
from vanguard_copilot.final_state_summary import load_final_state_summary
from vanguard_copilot.five_asset_hubo import load_five_asset_snapshot
from vanguard_copilot.higher_moment_extension import load_source_snapshot
from vanguard_copilot.hubo_quadratization import AncillaBudgetError,compare_qubo_with_hubo,export_qubo_upper_triangle,polynomial_degree,quadratize_portfolio_hubo
from vanguard_copilot.source_hubo_models import SourceHuboConfig
st.set_page_config(page_title="HUBO to QUBO Quadratization",page_icon="🔧",layout="wide"); st.title("Exact HUBO → QUBO Quadratization"); st.caption("Higher-order portfolio objective → reusable product ancillas → certified quadratic QUBO → exact ground-state audit.")
def _learned_config(qubits):
    summary=load_final_state_summary()
    if not summary:return None,"No saved automatic-HUBO final-state summary was found."
    if f"{qubits} qubits" not in str(summary.get("model","")):return None,"Latest saved summary is for a different model."
    l=summary.get("selected_lambdas") or {}; required={"return","variance","skewness","kurtosis","budget"}
    if not required.issubset(l):return None,"Saved summary does not contain all learned coefficients."
    return SourceHuboConfig(mode="budget_aligned",lambda_return=float(l["return"]),lambda_variance=float(l["variance"]),lambda_skewness=float(l["skewness"]),lambda_kurtosis=float(l["kurtosis"]),lambda_budget=float(l["budget"])),None
with st.sidebar:
    qubits=st.radio("HUBO size",[12,15],horizontal=True); use_learned=st.checkbox("Use latest learned λ*",value=True); cap_ancillas=st.checkbox("Impose ancilla budget",False); max_ancillas=st.number_input("Maximum ancillas",0,500,40,1,disabled=not cap_ancillas); safety=st.slider("Penalty safety factor",1.01,2.0,1.05,0.01); enumeration_limit=st.slider("Full-QUBO enumeration limit",12,26,22,1); run=st.button("Build and audit QUBO",type="primary",use_container_width=True)
if not run:st.info("Build the QUBO to measure ancilla overhead and verify exact energy/ground-state equivalence.");st.stop()
data=load_source_snapshot() if qubits==12 else load_five_asset_snapshot(); config=SourceHuboConfig(mode="budget_aligned"); source="Current fixed coefficients"
if use_learned:
    learned,warning=_learned_config(int(qubits))
    if learned is not None:config=learned;source="Latest validation-selected λ*"
    else:st.warning(warning+" Falling back to fixed coefficients.")
try:quad=quadratize_portfolio_hubo(data,config,max_ancillas=int(max_ancillas) if cap_ancillas else None,penalty_safety_factor=float(safety))
except AncillaBudgetError as exc:st.error(str(exc));st.stop()
audit=compare_qubo_with_hubo(data,config,quad,max_full_enumeration_variables=int(enumeration_limit));st.success("Exact quadratization completed.");st.caption(f"Coefficient source: {source}")
c=st.columns(8);c[0].metric("Original variables",quad.n_original_variables);c[1].metric("Ancillas",quad.n_ancillas);c[2].metric("Total QUBO variables",quad.n_total_variables);c[3].metric("HUBO degree",audit["original_hubo_degree"]);c[4].metric("QUBO degree",audit["qubo_degree"]);c[5].metric("Penalty P",f"{quad.penalty_strength:.4g}");c[6].metric("Range bound",f"{quad.objective_range_bound:.4g}");c[7].metric("Ancilla overhead",f"{100*quad.n_ancillas/quad.n_original_variables:.0f}%")
st.subheader("Exactness audit"); rows=[{"Check":"Source HUBO = explicit polynomial","Pass":audit["source_vs_polynomial_max_abs_error"]<=1e-8,"Error":audit["source_vs_polynomial_max_abs_error"]},{"Check":"HUBO = lifted QUBO for every original state","Pass":audit["energy_equivalence_passed"],"Error":audit["hubo_vs_lifted_qubo_max_abs_error"]},{"Check":"All lifted ancillas satisfy y=ab","Pass":audit["lifted_constraints_all_satisfied"],"Error":0.0},{"Check":"Lifted QUBO ground = HUBO ground","Pass":audit["lifted_ground_matches_hubo"],"Error":0.0}];st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
if audit["full_qubo_enumeration_performed"]:st.success(f"Full expanded-QUBO enumeration passed: {audit['full_qubo_states_enumerated']:,} states; ground match={audit['full_qubo_ground_matches_hubo']}.")
else:st.info(audit["full_qubo_enumeration_skip_reason"])
st.subheader("Penalty certification");st.latex(r"P>\Delta E_{objective},\quad C_{ab}=P(x_ax_b-2x_ay-2x_by+3y)");st.write("The penalty is chosen above a rigorous objective-range bound, preventing an ancilla-inconsistent assignment from becoming the global QUBO ground state.")
st.subheader("Polynomial complexity"); degree_counts_original={d:sum(1 for m in quad.original_polynomial if len(m)==d) for d in range(polynomial_degree(quad.original_polynomial)+1)};degree_counts_qubo={d:sum(1 for m in quad.qubo_polynomial if len(m)==d) for d in range(polynomial_degree(quad.qubo_polynomial)+1)}; complexity=pd.DataFrame([{"Degree":d,"Original HUBO":degree_counts_original.get(d,0),"Quadratized QUBO":degree_counts_qubo.get(d,0)} for d in sorted(set(degree_counts_original)|set(degree_counts_qubo))]);st.dataframe(complexity,hide_index=True,use_container_width=True);st.plotly_chart(px.bar(complexity.melt(id_vars="Degree",var_name="Representation",value_name="Term count"),x="Degree",y="Term count",color="Representation",barmode="group"),use_container_width=True)
with st.expander(f"Reusable ancilla map ({quad.n_ancillas} ancillas)"):
    st.dataframe(pd.DataFrame([{"Ancilla":c.ancilla,"Left":c.left,"Right":c.right,"Constraint":f"x{c.ancilla}=x{c.left}x{c.right}"} for c in quad.constraints]),hide_index=True,use_container_width=True)
with st.expander("Largest QUBO coefficients"):
    upper=export_qubo_upper_triangle(quad); terms=sorted(upper.items(),key=lambda item:abs(item[1]),reverse=True)[:100];st.dataframe(pd.DataFrame([{"i":p[0],"j":p[1],"Coefficient":v} for p,v in terms]),hide_index=True,use_container_width=True)
st.caption("Quadratization trades higher-order interactions for extra binary variables. Exact equivalence is audited; no quantum-advantage claim is made.")
