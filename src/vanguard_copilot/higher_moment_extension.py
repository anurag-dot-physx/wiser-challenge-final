"""Exploratory source-derived higher-moment HUBO/VQE appendix."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize

@dataclass(frozen=True)
class SourceMomentData:
    tickers: Tuple[str, ...]
    latest_prices: np.ndarray
    expected_returns: np.ndarray
    covariance: np.ndarray
    co_skewness: np.ndarray
    co_kurtosis: np.ndarray

@dataclass(frozen=True)
class HigherMomentConfig:
    total_units: int = 8
    bits_per_asset: int = 3
    lambda_return: float = 1.0
    lambda_variance: float = 2.0
    lambda_skewness: float = 0.25
    lambda_kurtosis: float = 0.10
    budget_penalty: float = 20.0
    risk_free_rate: float = 0.02
    layers: int = 2
    warmstart_epsilon: float = 0.10
    @property
    def allocation_step(self) -> float: return 1.0 / self.total_units

@dataclass(frozen=True)
class HigherMomentVqeResult:
    status: str
    selected_bitstring: str
    selected_weights: np.ndarray
    selected_metrics: Mapping[str, float]
    angles: np.ndarray
    runtime_seconds: float
    metadata: Mapping[str, Any]

def default_snapshot_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "source_portfolio_snapshot.json"

def load_source_snapshot(path: str | Path | None = None) -> SourceMomentData:
    source = default_snapshot_path() if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8")); tickers = tuple(str(v) for v in payload["tickers"])
    data = SourceMomentData(tickers, np.asarray(payload["latest_prices"], float), np.asarray(payload["exp_returns"], float), np.asarray(payload["cov_matrix"], float), np.asarray(payload["co_skewness"], float), np.asarray(payload["co_kurtosis"], float))
    n = len(tickers)
    if n != 4: raise ValueError("The exploratory source snapshot must contain four assets.")
    if data.latest_prices.shape != (n,) or data.expected_returns.shape != (n,): raise ValueError("Malformed source snapshot vectors.")
    if data.covariance.shape != (n,n): raise ValueError("Malformed covariance matrix.")
    if data.co_skewness.shape != (n,n,n): raise ValueError("Malformed co-skewness tensor.")
    if data.co_kurtosis.shape != (n,n,n,n): raise ValueError("Malformed co-kurtosis tensor.")
    return data

def bit_matrix(n_qubits: int) -> np.ndarray:
    states = np.arange(2**n_qubits, dtype=np.uint32); shifts = np.arange(n_qubits, dtype=np.uint32)
    return ((states[:,None] >> shifts[None,:]) & 1).astype(np.int8)

def units_from_bits(bits: np.ndarray, n_assets: int, bits_per_asset: int) -> np.ndarray:
    values = np.asarray(bits, dtype=int); single = values.ndim == 1; matrix = values[None,:] if single else values; powers = (2 ** np.arange(bits_per_asset)).astype(int); units = np.empty((matrix.shape[0], n_assets), dtype=int)
    for asset in range(n_assets): units[:,asset] = matrix[:, asset*bits_per_asset:(asset+1)*bits_per_asset] @ powers
    return units[0] if single else units

def source_bitstring(units: Sequence[int]) -> str: return "".join(format(int(v), "03b") for v in units)
def _normalized_tensor(tensor: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(tensor))); return np.zeros_like(tensor, dtype=float) if scale <= 1e-15 else tensor/scale

def state_table(data: SourceMomentData, config: HigherMomentConfig) -> Dict[str,np.ndarray]:
    n_assets = len(data.tickers); n_qubits = n_assets*config.bits_per_asset; bits = bit_matrix(n_qubits); units = units_from_bits(bits,n_assets,config.bits_per_asset); weights = units.astype(float)/config.total_units; total_units=units.sum(axis=1); feasible=total_units==config.total_units
    expected_return = weights @ data.expected_returns; variance=np.einsum("ni,ij,nj->n",weights,data.covariance,weights,optimize=True); volatility=np.sqrt(np.maximum(variance,0.0))
    skewness=np.einsum("ni,nj,nk,ijk->n",weights,weights,weights,_normalized_tensor(data.co_skewness),optimize=True); kurtosis=np.einsum("ni,nj,nk,nl,ijkl->n",weights,weights,weights,weights,_normalized_tensor(data.co_kurtosis),optimize=True)
    mean_variance=-config.lambda_return*expected_return+config.lambda_variance*variance; tail_aware=mean_variance-config.lambda_skewness*skewness+config.lambda_kurtosis*kurtosis; hubo_energy=tail_aware+config.budget_penalty*(total_units-config.total_units)**2; sharpe=(expected_return-config.risk_free_rate)/np.maximum(volatility,1e-12)
    return {"bits":bits,"units":units,"weights":weights,"total_units":total_units,"feasible":feasible,"expected_return":expected_return,"variance":variance,"volatility":volatility,"sharpe_ratio":sharpe,"normalized_skewness":skewness,"normalized_kurtosis":kurtosis,"mean_variance_energy":mean_variance,"tail_aware_energy":tail_aware,"hubo_energy":hubo_energy}

def _state_record(table: Mapping[str,np.ndarray], index:int)->Dict[str,Any]:
    units=table["units"][index]
    return {"state_index":int(index),"bitstring":source_bitstring(units),"units":[int(v) for v in units],"weights":[float(v) for v in table["weights"][index]],"expected_return":float(table["expected_return"][index]),"variance":float(table["variance"][index]),"volatility":float(table["volatility"][index]),"sharpe_ratio":float(table["sharpe_ratio"][index]),"normalized_skewness":float(table["normalized_skewness"][index]),"normalized_kurtosis":float(table["normalized_kurtosis"][index]),"mean_variance_energy":float(table["mean_variance_energy"][index]),"tail_aware_energy":float(table["tail_aware_energy"][index]),"hubo_energy":float(table["hubo_energy"][index]),"fully_invested":bool(table["feasible"][index])}

def exact_higher_moment_benchmark(data:SourceMomentData|None=None,config:HigherMomentConfig|None=None)->Dict[str,Any]:
    data=load_source_snapshot() if data is None else data; config=HigherMomentConfig() if config is None else config; table=state_table(data,config); feasible_indices=np.flatnonzero(table["feasible"]); mv_index=int(feasible_indices[np.argmin(table["mean_variance_energy"][feasible_indices])]); tail_index=int(feasible_indices[np.argmin(table["tail_aware_energy"][feasible_indices])]); hubo_index=int(np.argmin(table["hubo_energy"]))
    return {"scope":"exploratory four-equity higher-moment benchmark","claim_boundary":"This extension is separate from the eight-asset production co-pilot and does not replace the flagship constrained QUBO results.","assets":list(data.tickers),"n_qubits":len(data.tickers)*config.bits_per_asset,"allocation_step":config.allocation_step,"states_enumerated":int(table["bits"].shape[0]),"feasible_states":int(table["feasible"].sum()),"configuration":dict(config.__dict__),"mean_variance_optimum":_state_record(table,mv_index),"higher_moment_optimum":_state_record(table,tail_index),"hubo_ground_state":_state_record(table,hubo_index),"hubo_ground_state_matches_higher_moment_optimum":bool(hubo_index==tail_index),"interpretation":{"return_change":float(table["expected_return"][tail_index]-table["expected_return"][mv_index]),"volatility_change":float(table["volatility"][tail_index]-table["volatility"][mv_index]),"skewness_change":float(table["normalized_skewness"][tail_index]-table["normalized_skewness"][mv_index]),"kurtosis_change":float(table["normalized_kurtosis"][tail_index]-table["normalized_kurtosis"][mv_index])}}

def legacy_source_diagnostic(data:SourceMomentData|None=None,*,total_budget:float=10000.0,dollar_unit:float=1150.0)->Dict[str,Any]:
    data=load_source_snapshot() if data is None else data; bits=bit_matrix(12); units=units_from_bits(bits,4,3); encoded_spend=units.sum(axis=1)*dollar_unit; breach=np.abs(encoded_spend-total_budget); minimum=float(np.min(breach)); candidates=np.flatnonzero(np.isclose(breach,minimum,atol=1e-9)); weights=units[candidates].astype(float); weights/=weights.sum(axis=1,keepdims=True); returns=weights@data.expected_returns; variance=np.einsum("ni,ij,nj->n",weights,data.covariance,weights,optimize=True); sharpe=(returns-0.02)/np.sqrt(np.maximum(variance,1e-12)); local=int(np.argmax(sharpe)); selected=int(candidates[local]); selected_units=units[selected]; dollars=selected_units.astype(float)*dollar_unit; shares=np.floor(dollars/data.latest_prices).astype(int); actual_spend=float(shares@data.latest_prices)
    return {"description":"Diagnostic reproduction of the original $1,150 encoding","target_budget":total_budget,"dollar_unit":dollar_unit,"target_units":total_budget/dollar_unit,"exact_encoded_budget_representable":bool(np.isclose(total_budget/dollar_unit,round(total_budget/dollar_unit))),"minimum_encoded_breach":minimum,"financial_truth_bitstring":source_bitstring(selected_units),"financial_truth_units":[int(v) for v in selected_units],"encoded_spend":float(encoded_spend[selected]),"source_style_sharpe":float(sharpe[local]),"whole_shares":{ticker:int(v) for ticker,v in zip(data.tickers,shares)},"actual_cash_spent":actual_spend,"actual_cash_reserve":float(total_budget-actual_spend),"warning":"The source's nearest encoded budget and final whole-share expenditure are different quantities; this diagnostic is not used by the flagship solver."}

def _apply_ry(state:np.ndarray,theta:float,qubit:int)->None:
    c=float(np.cos(theta/2.0)); s=float(np.sin(theta/2.0)); stride=1<<qubit; block=stride<<1
    for start in range(0,state.size,block):
        left=slice(start,start+stride); right=slice(start+stride,start+block); a=state[left].copy(); b=state[right].copy(); state[left]=c*a-s*b; state[right]=s*a+c*b

def _apply_cnot(state:np.ndarray,control:int,target:int)->None:
    control_mask=1<<control; target_mask=1<<target
    for index in range(state.size):
        if (index&control_mask) and not (index&target_mask):
            partner=index|target_mask; state[index],state[partner]=state[partner],state[index]

def vqe_state(params:np.ndarray,n_qubits:int,layers:int)->np.ndarray:
    angles=np.asarray(params,float).reshape(layers+1,n_qubits); state=np.zeros(2**n_qubits,float); state[0]=1.0
    for q in range(n_qubits): _apply_ry(state,angles[0,q],q)
    for layer in range(layers):
        for q in range(n_qubits-1): _apply_cnot(state,q,q+1)
        _apply_cnot(state,n_qubits-1,0)
        for q in range(n_qubits): _apply_ry(state,angles[layer+1,q],q)
    norm=float(np.linalg.norm(state));
    if norm<=0.0: raise RuntimeError("VQE state has zero norm.")
    return state/norm

def _equal_weight_warm_start(config:HigherMomentConfig,n_assets:int)->np.ndarray:
    units=np.full(n_assets,config.total_units//n_assets,dtype=int); target_bits=[(unit>>bit)&1 for unit in units for bit in range(config.bits_per_asset)]; eps=config.warmstart_epsilon; first=np.array([2.0*np.arcsin(np.sqrt(1.0-eps if bit else eps)) for bit in target_bits],float); params=np.zeros((config.layers+1,n_assets*config.bits_per_asset),float); params[0]=first; return params.ravel()

def run_higher_moment_vqe(data:SourceMomentData|None=None,config:HigherMomentConfig|None=None,*,maxiter:int=80,shots:int=4096,seed:int=42,restarts:int=2)->HigherMomentVqeResult:
    if maxiter<=0 or shots<=0 or restarts<=0: raise ValueError("maxiter, shots, and restarts must be positive.")
    started=perf_counter(); data=load_source_snapshot() if data is None else data; config=HigherMomentConfig() if config is None else config; table=state_table(data,config); energies=np.asarray(table["hubo_energy"],float); feasible=np.asarray(table["feasible"],bool); scale=max(float(np.std(energies)),1e-12); normalized=(energies-float(np.mean(energies)))/scale; n_qubits=len(data.tickers)*config.bits_per_asset; rng=np.random.default_rng(seed)
    def expectation(params):
        state=vqe_state(params,n_qubits,config.layers); return float((state*state)@normalized)
    best=None
    for restart in range(restarts):
        initial=_equal_weight_warm_start(config,len(data.tickers)) if restart==0 else rng.uniform(-np.pi,np.pi,size=(config.layers+1)*n_qubits); result=minimize(expectation,initial,method="COBYLA",options={"maxiter":maxiter,"rhobeg":0.25,"tol":1e-5});
        if best is None or float(result.fun)<float(best.fun): best=result
    assert best is not None
    state=vqe_state(best.x,n_qubits,config.layers); probabilities=state*state; probabilities/=probabilities.sum(); counts=rng.multinomial(shots,probabilities); observed=np.flatnonzero(counts>0); feasible_observed=observed[feasible[observed]]; exact=exact_higher_moment_benchmark(data,config); exact_index=int(exact["higher_moment_optimum"]["state_index"]); fallback=feasible_observed.size==0
    if fallback: selected_index=exact_index; selected_count=0; status="completed_with_exact_classical_fallback"
    else: local=int(np.argmin(table["tail_aware_energy"][feasible_observed])); selected_index=int(feasible_observed[local]); selected_count=int(counts[selected_index]); status="completed"
    record=_state_record(table,selected_index)
    if not record["fully_invested"]: raise RuntimeError("Internal error: VQE reported an infeasible portfolio.")
    metric_keys=("expected_return","variance","volatility","sharpe_ratio","normalized_skewness","normalized_kurtosis","mean_variance_energy","tail_aware_energy","hubo_energy"); metrics={k:float(record[k]) for k in metric_keys}
    return HigherMomentVqeResult(status,str(record["bitstring"]),np.asarray(record["weights"],float),metrics,np.asarray(best.x,float),perf_counter()-started,{"maxiter":maxiter,"shots":shots,"restarts":restarts,"optimizer_status":str(best.message),"expectation_normalized":float(best.fun),"feasible_probability_mass":float(probabilities[feasible].sum()),"observed_unique_states":int(observed.size),"observed_feasible_states":int(feasible_observed.size),"selected_state_index":selected_index,"selected_observed_count":selected_count,"fully_invested":True,"exact_higher_moment_probability":float(probabilities[exact_index]),"exact_higher_moment_recovered":bool(selected_index==exact_index),"optimality_gap":float(table["tail_aware_energy"][selected_index]-table["tail_aware_energy"][exact_index]),"classical_fallback_used":fallback})

def extension_report(*,run_vqe:bool=False,maxiter:int=80,shots:int=4096,seed:int=42)->Dict[str,Any]:
    data=load_source_snapshot(); config=HigherMomentConfig(); report={"scope_note":"Exploratory source-derived four-equity higher-moment appendix; separate from the flagship eight-asset production model.","exact_benchmark":exact_higher_moment_benchmark(data,config),"legacy_source_diagnostic":legacy_source_diagnostic(data)}
    if run_vqe:
        result=run_higher_moment_vqe(data,config,maxiter=maxiter,shots=shots,seed=seed); report["vqe"]={"status":result.status,"selected_bitstring":result.selected_bitstring,"weights":{ticker:float(weight) for ticker,weight in zip(data.tickers,result.selected_weights)},"selected_metrics":dict(result.selected_metrics),"angles":result.angles.tolist(),"runtime_seconds":float(result.runtime_seconds),"metadata":dict(result.metadata)}
    return report
