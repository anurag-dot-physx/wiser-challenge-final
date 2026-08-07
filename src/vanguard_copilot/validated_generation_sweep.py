"""Validation-selected generation sweep built on one cached CMA-ES trajectory."""
from __future__ import annotations
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from .generation_selection import SELECTION_OBJECTIVES, _aggregate, _evaluate_candidate, build_evaluation_cache, train_generation_checkpoints
from .higher_moment_extension import SourceMomentData
from .source_hubo_models import SourceHuboConfig

def _pareto_flags(candidates: Sequence[Mapping[str, Any]]) -> list[bool]:
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

def _select_index(candidates, objective, tolerance):
    if objective not in SELECTION_OBJECTIVES: raise ValueError(f"Unknown selection objective: {objective}")
    feasible=np.asarray([c["validation_feasibility_rate"]>=1.0-1e-12 for c in candidates]); pool=np.flatnonzero(feasible)
    if pool.size==0:
        rates=np.asarray([c["validation_feasibility_rate"] for c in candidates]); pool=np.flatnonzero(np.isclose(rates,rates.max()))
    if objective=="highest_sharpe": return int(pool[np.argmax([candidates[i]["validation_mean_sharpe"] for i in pool])])
    if objective=="highest_return": return int(pool[np.argmax([candidates[i]["validation_mean_return"] for i in pool])])
    if objective=="lowest_volatility": return int(pool[np.argmin([candidates[i]["validation_mean_volatility"] for i in pool])])
    sharpes=np.asarray([candidates[i]["validation_mean_sharpe"] for i in pool],float); best=float(sharpes.max()); band=float(tolerance)*max(1.0,abs(best)); eligible=[int(i) for i in pool if candidates[int(i)]["validation_mean_sharpe"]>=best-band]; return min(eligible,key=lambda i:(candidates[i]["generation"],candidates[i]["training_elapsed_seconds"]))

def validated_generation_sweep(train_data:SourceMomentData,validation_data:Sequence[SourceMomentData],held_out_data:Sequence[SourceMomentData],*,validation_names:Sequence[str],held_out_names:Sequence[str],checkpoints:Sequence[int]=(5,10,20,30,50,75,100),population_size:int=12,seed:int=42,objective:str="efficient_sharpe",near_optimal_tolerance:float=0.01)->Dict[str,Any]:
    started=perf_counter(); trajectory=train_generation_checkpoints(train_data,checkpoints,population_size=population_size,seed=seed); val_caches=[build_evaluation_cache(data) for data in validation_data]; test_caches=[build_evaluation_cache(data) for data in held_out_data]; candidates=[]
    for checkpoint in trajectory["checkpoints"]:
        lambdas=np.asarray(checkpoint["lambda_vector"],float); details=[_evaluate_candidate(lambdas,cache) for cache in val_caches]; aggregate=_aggregate(details); row=dict(checkpoint); row.update({"validation_mean_return":aggregate["mean_return"],"validation_mean_volatility":aggregate["mean_volatility"],"validation_mean_sharpe":aggregate["mean_sharpe"],"validation_mean_sharpe_abs_error":aggregate["mean_sharpe_abs_error"],"validation_feasibility_rate":aggregate["feasibility_rate"],"validation_mean_budget_breach":aggregate["mean_budget_breach"],"validation_details":[dict(name=name,**detail) for name,detail in zip(validation_names,details)]}); candidates.append(row)
    for row,flag in zip(candidates,_pareto_flags(candidates)): row["pareto_efficient"]=bool(flag)
    selected_index=_select_index(candidates,objective,near_optimal_tolerance); selected=candidates[selected_index]; selected_lambdas=np.asarray(selected["lambda_vector"],float); held_details=[_evaluate_candidate(selected_lambdas,cache) for cache in test_caches]; held_aggregate=_aggregate(held_details)
    return {"status":"completed","objective":objective,"objective_label":SELECTION_OBJECTIVES[objective],"candidate_generations":[int(c["generation"]) for c in candidates],"candidates":candidates,"selected_generation":int(selected["generation"]),"selected_lambda_vector":[float(x) for x in selected_lambdas],"selected_lambdas":dict(selected["lambdas"]),"selected_validation_metrics":{"mean_return":float(selected["validation_mean_return"]),"mean_volatility":float(selected["validation_mean_volatility"]),"mean_sharpe":float(selected["validation_mean_sharpe"]),"mean_sharpe_abs_error":float(selected["validation_mean_sharpe_abs_error"]),"feasibility_rate":float(selected["validation_feasibility_rate"]),"mean_budget_breach":float(selected["validation_mean_budget_breach"])},"held_out_metrics":held_aggregate,"held_out_details":[dict(name=name,**detail) for name,detail in zip(held_out_names,held_details)],"training_evaluations":int(trajectory["evaluations"]),"training_seconds":float(trajectory["training_seconds"]),"total_selection_seconds":float(perf_counter()-started),"trajectory":trajectory}

def config_from_validated_sweep(sweep:Mapping[str,Any])->SourceHuboConfig:
    v=sweep["selected_lambdas"]; return SourceHuboConfig(mode="budget_aligned",lambda_return=float(v["return"]),lambda_variance=float(v["variance"]),lambda_skewness=float(v["skewness"]),lambda_kurtosis=float(v["kurtosis"]),lambda_budget=float(v["budget"]))
