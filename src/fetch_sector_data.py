"""Fetch and vectorize the 1-train + 10-sector higher-moment dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
TRAIN_TICKERS=["AAPL","MSFT","NVDA","AMZN"]
TEST_PORTFOLIOS={"test_0":["JPM","BAC","WFC","C"],"test_1":["JNJ","PFE","UNH","LLY"],"test_2":["XOM","CVX","COP","SLB"],"test_3":["PG","KO","PEP","COST"],"test_4":["HD","NKE","MCD","SBUX"],"test_5":["CAT","DE","GE","MMM"],"test_6":["GOOGL","META","NFLX","DIS"],"test_7":["AMT","PLD","CCI","EQIX"],"test_8":["NEE","DUK","SO","AEP"],"test_9":["V","MA","PYPL","AXP"]}; START_DATE="2021-01-01"; END_DATE="2024-01-01"
def _moments(prices,tickers):
    frame=prices[tickers].dropna(how="any"); daily=frame.pct_change(fill_method=None).dropna().to_numpy(float); latest=frame.iloc[-1].to_numpy(float); mean_daily=daily.mean(axis=0); centered=daily-mean_daily; n_obs=float(centered.shape[0]); exp_returns=mean_daily*252.0; covariance=np.cov(daily,rowvar=False,ddof=1)*252.0; co_skewness=np.einsum("ti,tj,tk->ijk",centered,centered,centered,optimize=True)*(252.0**1.5)/n_obs; co_kurtosis=np.einsum("ti,tj,tk,tl->ijkl",centered,centered,centered,centered,optimize=True)*(252.0**2)/n_obs; return {"tickers":np.asarray(tickers),"latest_prices":latest,"exp_returns":exp_returns,"cov_matrix":covariance,"co_skewness":co_skewness,"co_kurtosis":co_kurtosis}
def main():
    all_tickers=list(dict.fromkeys(TRAIN_TICKERS+[x for group in TEST_PORTFOLIOS.values() for x in group])); raw=yf.download(all_tickers,start=START_DATE,end=END_DATE,auto_adjust=True,progress=False)
    if raw.empty:raise RuntimeError("No market data returned by yfinance.")
    if isinstance(raw.columns,pd.MultiIndex):close=raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close",axis=1,level=0)
    else:close=raw
    close=close.reindex(columns=all_tickers); payload={}; groups={"train":TRAIN_TICKERS,**TEST_PORTFOLIOS}
    for name,tickers in groups.items():
        for key,value in _moments(close,tickers).items():payload[f"{name}_{key}"]=value
    output=Path(__file__).resolve().parents[1]/"data"/"portfolio_data_2.npz";output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**payload);print(f"Saved {len(groups)} datasets to {output}")
if __name__=="__main__":main()
