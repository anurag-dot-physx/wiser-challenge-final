"""Leakage-safe generation sweep and validation selection for the full-tensor HUBO."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .exact_lambda_training import LAMBDA_NAMES, LOG10_BOUNDS, SOURCE_WARM_START, TrainingConfig, _population_losses, build_feature_cache
from .higher_moment_extension import SourceMomentData
from .source_hubo_models import SourceHuboConfig, source_hubo_state_table

SELECTION_OBJECTIVES={"highest_sharpe":"Highest validation Sharpe","highest_return":"Highest validation return","lowest_volatility":"Lowest validation volatility","efficient_sharpe":"Fastest near-optimal validation Sharpe"}

@dataclass(frozen=True)
class EvaluationCache:
    features: np.ndarray
    expected_return: np.ndarray
    volatility: np.ndarray
    sharpe: np.ndarray
    budget_breach: np.ndarray
    feasible: np.ndarray
    financial_index: int

def build_evaluation_cache(data:SourceMomentData)->EvaluationCache:
    cfg=SourceHuboConfig(mode="budget_aligned",lambda_return=1.0,lambda_variance=1.0,lambda_skewness=1.0,lambda_kurtosis=1.0,lambda_budget=1.0); table=source_hubo_state_table(data,cfg)
    features=np.column_stack([table["return_energy"],table["variance_energy"],table["skewness_energy"],table["kurtosis_energy"],table["budget_energy"]]).astype(float,copy=False); feasible=np.asarray(table["admissible"],bool); feasible_idx=np.flatnonzero(feasible); financial_idx=int(feasible_idx[np.argmax(table["sharpe_ratio"][feasible_idx])])
    return EvaluationCache(np.ascontiguousarray(features),np.asarray(table["expected_return"],float),np.asarray(table["volatility"],float),np.asarray(table["sharpe_ratio"],float),np.asarray(table["budget_breach"],float),feasible,financial_idx)

def _snapshot(best_log,generation,loss,ground,elapsed,train_cache):
    lambdas=np.power(10.0,best_log); return {"generation":int(generation),"lambda_vector":[float(x) for x in lambdas],"lambdas":{name:float(value) for name,value in zip(LAMBDA_NAMES,lambdas)},"training_loss":float(loss),"training_ground_state_index":int(ground),"training_ground_feasible":bool(train_cache.feasible[ground]),"training_ground_sharpe":float(train_cache.sharpe[ground]),"training_elapsed_seconds":float(elapsed)}

def train_generation_checkpoints(data:SourceMomentData,checkpoints:Sequence[int],*,population_size:int=12,seed:int=42,initial_sigma:float=0.45)->Dict[str,Any]:
    ordered=sorted({int(x) for x in checkpoints if int(x)>0});
    if not ordered: raise ValueError("At least one positive generation checkpoint is required.")
    if population_size<4: raise ValueError("population_size must be >= 4")
    started=perf_counter(); cfg=TrainingConfig(generations=max(ordered),population_size=int(population_size),seed=int(seed),initial_sigma=float(initial_sigma),patience=max(ordered)+1); cache=build_feature_cache(data); n=len(LAMBDA_NAMES); popsize=int(population_size); mu=popsize//2; raw_weights=np.log(mu+0.5)-np.log(np.arange(1,mu+1)); weights=raw_weights/raw_weights.sum(); mueff=float(1.0/np.sum(weights**2)); cc=(4.0+mueff/n)/(n+4.0+2.0*mueff/n); cs=(mueff+2.0)/(n+mueff+5.0); c1=2.0/((n+1.3)**2+mueff); cmu=min(1.0-c1,2.0*(mueff-2.0+1.0/mueff)/((n+2.0)**2+mueff)); damps=1.0+2.0*max(0.0,np.sqrt((mueff-1.0)/(n+1.0))-1.0)+cs; chi_n=np.sqrt(n)*(1.0-1.0/(4.0*n)+1.0/(21.0*n*n))
    rng=np.random.default_rng(seed); mean=np.log10(SOURCE_WARM_START); sigma=float(initial_sigma); cov=np.eye(n); pc=np.zeros(n); ps=np.zeros(n); warm_loss,warm_ground,_=_population_losses(mean[None,:],cache,cfg); best_loss=float(warm_loss[0]); best_log=mean.copy(); best_ground=int(warm_ground[0]); evaluations=1; snapshots={}; trace=[]
    for generation in range(1,max(ordered)+1):
        eigvals,eigvecs=np.linalg.eigh(cov); eigvals=np.maximum(eigvals,1e-12); transform=eigvecs@np.diag(np.sqrt(eigvals)); inv_sqrt=eigvecs@np.diag(1.0/np.sqrt(eigvals))@eigvecs.T; z=rng.standard_normal((popsize,n)); y=z@transform.T; candidates=np.clip(mean[None,:]+sigma*y,LOG10_BOUNDS[:,0],LOG10_BOUNDS[:,1]); y=(candidates-mean[None,:])/max(sigma,1e-12); losses,grounds,_=_population_losses(candidates,cache,cfg); evaluations+=popsize; order=np.argsort(losses); elite=order[:mu]; generation_best=int(order[0])
        if float(losses[generation_best])<best_loss-1e-12: best_loss=float(losses[generation_best]); best_log=candidates[generation_best].copy(); best_ground=int(grounds[generation_best])
        old_mean=mean.copy(); mean=np.sum(weights[:,None]*candidates[elite],axis=0); y_w=(mean-old_mean)/max(sigma,1e-12); ps=(1.0-cs)*ps+np.sqrt(cs*(2.0-cs)*mueff)*(inv_sqrt@y_w); ps_norm=float(np.linalg.norm(ps)); hsig=float(ps_norm/np.sqrt(max(1e-15,1.0-(1.0-cs)**(2.0*generation)))<(1.4+2.0/(n+1.0))*chi_n); pc=(1.0-cc)*pc+hsig*np.sqrt(cc*(2.0-cc)*mueff)*y_w; rank_mu=np.zeros((n,n))
        for weight,step in zip(weights,y[elite]): rank_mu+=weight*np.outer(step,step)
        cov=(1.0-c1-cmu)*cov+c1*(np.outer(pc,pc)+(1.0-hsig)*cc*(2.0-cc)*cov)+cmu*rank_mu; cov=0.5*(cov+cov.T); sigma*=float(np.exp((cs/damps)*(ps_norm/chi_n-1.0))); sigma=float(np.clip(sigma,0.03,1.2)); elapsed=perf_counter()-started; trace.append({"generation":generation,"global_best_loss":best_loss,"ground_state_index":best_ground,"ground_feasible":bool(cache.feasible[best_ground]),"ground_sharpe":float(cache.sharpe[best_ground]),"sigma":sigma,"elapsed_seconds":elapsed})
        if generation in ordered: snapshots[generation]=_snapshot(best_log,generation,best_loss,best_ground,elapsed,cache)
        if best_ground==cache.financial_ground_truth_index:
            for checkpoint in ordered:
                if checkpoint>generation: snapshots[checkpoint]=_snapshot(best_log,checkpoint,best_loss,best_ground,elapsed,cache)
            break
    return {"status":"completed","method":"single-trajectory exact-ground-state CMA-ES checkpoint sweep","checkpoints":[snapshots[g] for g in ordered],"trace":trace,"evaluations":int(evaluations),"training_seconds":float(perf_counter()-started),"state_count":int(cache.state_count),"financial_ground_truth_index":int(cache.financial_ground_truth_index),"financial_ground_truth_sharpe":float(cache.financial_ground_truth_sharpe)}

def _evaluate_candidate(lambdas:np.ndarray,cache:EvaluationCache)->Dict[str,Any]:
    energies=cache.features@lambdas; idx=int(np.argmin(energies)); financial=cache.financial_index
    return {"ground_state_index":idx,"ground_feasible":bool(cache.feasible[idx]),"budget_breach":float(cache.budget_breach[idx]),"expected_return":float(cache.expected_return[idx]),"volatility":float(cache.volatility[idx]),"sharpe":float(cache.sharpe[idx]),"financial_sharpe":float(cache.sharpe[financial]),"sharpe_abs_error":float(abs(cache.sharpe[idx]-cache.sharpe[financial]))}

def _aggregate(rows:Sequence[Mapping[str,Any]])->Dict[str,float]:
    return {"mean_return":float(np.mean([x["expected_return"] for x in rows])),"mean_volatility":float(np.mean([x["volatility"] for x in rows])),"mean_sharpe":float(np.mean([x["sharpe"] for x in rows])),"mean_sharpe_abs_error":float(np.mean([x["sharpe_abs_error"] for x in rows])),"feasibility_rate":float(np.mean([bool(x["ground_feasible"]) for x in rows])),"mean_budget_breach":float(np.mean([x["budget_breach"] for x in rows]))}

def _pareto_flags(candidates):
    flags=[]
    for i,c in enumerate(candidates):
        dominated=False
        for j,o in enumerate(candidates):
            if i==j: continue
            weak=o["validation_mean_return"]>=c["validation_mean_return"] and o["validation_mean_volatility"]<=c["validation_mean_volatility"] and o["training_elapsed_seconds"]<=c["training_elapsed_seconds"]
            strict=o["validation_mean_return"]>c["validation_mean_return"] or o["validation_mean_volatility"]<c["validation_mean_volatility"] or o["training_elapsed_seconds"]<c["training_elapsed_seconds"]
            if weak and strict: dominated=True; break
        flags.append(not dominated)
    return flags

def _select_index(candidates,objective,near_optimal_tolerance):
    if objective not in SELECTION_OBJECTIVES: raise ValueError(f"Unknown selection objective: {objective}")
    feasible=np.asarray([x["validation_feasibility_rate"]>=1.0-1e-12 for x in candidates]); pool=np.flatnonzero(feasible)
    if pool.size==0: rates=np.asarray([x["validation_feasibility_rate"] for x in candidates]); pool=np.flatnonzero(np.isclose(rates,rates.max()))
    if objective=="highest_sharpe": return int(pool[np.argmax([candidates[i]["validation_mean_sharpe"] for i in pool])])
    if objective=="highest_return": return int(pool[np.argmax([candidates[i]["validation_mean_return"] for i in pool])])
    if objective=="lowest_volatility": return int(pool[np.argmin([candidates[i]["validation_mean_volatility"] for i in pool])])
    sharpes=np.asarray([candidates[i]["validation_mean_sharpe"] for i in pool]); best=float(np.max(sharpes)); tolerance=float(near_optimal_tolerance)*max(1.0,abs(best)); eligible=[int(i) for i in pool if candidates[int(i)]["validation_mean_sharpe"]>=best-tolerance]; return min(eligible,key=lambda i:(candidates[i]["generation"],candidates[i]["training_elapsed_seconds"]))

def generation_sweep_select(train_data,validation_data,held_out_data,*,validation_names=None,held_out_names=None,checkpoints=(5,10,20,30,50,75,100),population_size=12,seed=42,objective="efficient_sharpe",near_optimal_tolerance=0.01):
    started=perf_counter(); trajectory=train_generation_checkpoints(train_data,checkpoints,population_size=population_size,seed=seed); validation_caches=[build_evaluation_cache(d) for d in validation_data]; held_out_caches=[build_evaluation_cache(d) for d in held_out_data]; val_names=list(validation_names or [f"validation_{i}" for i in range(len(validation_caches))]); test_names=list(held_out_names or [f"held_out_{i}" for i in range(len(held_out_caches))]); candidates=[]
    for checkpoint in trajectory["checkpoints"]:
        lambdas=np.asarray(checkpoint["lambda_vector"],float); details=[_evaluate_candidate(lambdas,c) for c in validation_caches]; aggregate=_aggregate(details); row=dict(checkpoint); row.update({f"validation_{key[5:] if key.startswith('mean_') else key}":value for key,value in aggregate.items()}); row["validation_details"]=[dict(name=name,**detail) for name,detail in zip(val_names,details)]; candidates.append(row)
    for row,flag in zip(candidates,_pareto_flags(candidates)): row["pareto_efficient"]=bool(flag)
    selected_index=_select_index(candidates,objective,near_optimal_tolerance); selected=candidates[selected_index]; selected_lambdas=np.asarray(selected["lambda_vector"],float); held_details=[_evaluate_candidate(selected_lambdas,c) for c in held_out_caches]; held_aggregate=_aggregate(held_details)
    return {"status":"completed","objective":objective,"objective_label":SELECTION_OBJECTIVES[objective],"near_optimal_tolerance":float(near_optimal_tolerance),"candidate_generations":[int(x["generation"]) for x in candidates],"candidates":candidates,"selected_generation":int(selected["generation"]),"selected_lambda_vector":[float(x) for x in selected_lambdas],"selected_lambdas":dict(selected["lambdas"]),"selected_validation_metrics":{"mean_return":float(selected["validation_mean_return"]),"mean_volatility":float(selected["validation_mean_volatility"]),"mean_sharpe":float(selected["validation_mean_sharpe"]),"mean_sharpe_abs_error":float(selected["validation_sharpe_abs_error"]),"feasibility_rate":float(selected["validation_feasibility_rate"]),"mean_budget_breach":float(selected["validation_budget_breach"])},"held_out_metrics":held_aggregate,"held_out_details":[dict(name=name,**detail) for name,detail in zip(test_names,held_details)],"training_evaluations":int(trajectory["evaluations"]),"training_seconds":float(trajectory["training_seconds"]),"total_selection_seconds":float(perf_counter()-started),"trajectory":trajectory}

def source_hubo_config_from_selected_sweep(sweep:Mapping[str,Any])->SourceHuboConfig:
    values=sweep["selected_lambdas"]; return SourceHuboConfig(mode="budget_aligned",lambda_return=float(values["return"]),lambda_variance=float(values["variance"]),lambda_skewness=float(values["skewness"]),lambda_kurtosis=float(values["kurtosis"]),lambda_budget=float(values["budget"]))
