"""Five-asset, 15-qubit variants of the higher-moment HUBO extensions."""
from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict

import numpy as np

from .higher_moment_extension import HigherMomentConfig, SourceMomentData, exact_higher_moment_benchmark, run_higher_moment_vqe
from .source_hubo_models import SourceHuboConfig, exact_source_hubo_report, run_source_hubo_vqe

_FIVE_ASSET_PAYLOAD_ZLIB_B64 = (
    "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"
)

def load_five_asset_snapshot() -> SourceMomentData:
    payload = json.loads(zlib.decompress(base64.b64decode(_FIVE_ASSET_PAYLOAD_ZLIB_B64)).decode("utf-8")); tickers=tuple(str(v) for v in payload["tickers"])
    data=SourceMomentData(tickers,np.asarray(payload["latest_prices"],float),np.asarray(payload["exp_returns"],float),np.asarray(payload["cov_matrix"],float),np.asarray(payload["co_skewness"],float),np.asarray(payload["co_kurtosis"],float)); n=len(tickers)
    if n!=5: raise ValueError("The 15-qubit snapshot must contain exactly five assets.")
    if data.latest_prices.shape!=(n,) or data.expected_returns.shape!=(n,): raise ValueError("Malformed five-asset source vectors.")
    if data.covariance.shape!=(n,n): raise ValueError("Malformed five-asset covariance matrix.")
    if data.co_skewness.shape!=(n,n,n): raise ValueError("Malformed five-asset co-skewness tensor.")
    if data.co_kurtosis.shape!=(n,n,n,n): raise ValueError("Malformed five-asset co-kurtosis tensor.")
    return data

def _provenance(data: SourceMomentData) -> Dict[str,Any]:
    return {"assets":list(data.tickers),"note":"AAPL/MSFT/AMZN/NVDA joint moments come from the supplied training dataset; AAPL/MSFT/TSLA/AMZN joint moments come from the supplied original dataset. Terms containing both TSLA and NVDA were unavailable and are set to zero."}

def original_fifteen_qubit_report(*,run_vqe:bool=False,maxiter:int=80,shots:int=4096,seed:int=42)->Dict[str,Any]:
    data=load_five_asset_snapshot(); config=HigherMomentConfig(); report={"scope_note":"Five-asset, 15-qubit extension of the original normalized-moment HUBO/VQE.","exact_benchmark":exact_higher_moment_benchmark(data,config),"data_provenance":_provenance(data)}
    if run_vqe:
        result=run_higher_moment_vqe(data,config,maxiter=maxiter,shots=shots,seed=seed); report["vqe"]={"status":result.status,"selected_bitstring":result.selected_bitstring,"weights":{ticker:float(weight) for ticker,weight in zip(data.tickers,result.selected_weights)},"selected_metrics":dict(result.selected_metrics),"angles":result.angles.tolist(),"runtime_seconds":float(result.runtime_seconds),"metadata":dict(result.metadata)}
    return report

def budget_aligned_fifteen_qubit_report(*,run_vqe:bool=False,maxiter:int=80,shots:int=4096,seed:int=42)->Dict[str,Any]:
    data=load_five_asset_snapshot(); config=SourceHuboConfig(mode="budget_aligned"); report=exact_source_hubo_report(data,config); report["data_provenance"]=_provenance(data)
    if run_vqe: report["vqe"]=run_source_hubo_vqe(data,config,maxiter=maxiter,shots=shots,seed=seed)
    return report
