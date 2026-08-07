"""Load the supplied multi-sector dataset for validation and held-out evaluation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from .higher_moment_extension import SourceMomentData

VALIDATION_NAMES = ["Financials", "Healthcare", "Energy", "Consumer Staples", "Consumer Discretionary"]
HELD_OUT_NAMES = ["Industrials", "Communication Services", "Real Estate", "Utilities", "Payments"]


def _candidate_paths() -> List[Path]:
    paths: List[Path] = []
    env = os.environ.get("VANGUARD_PORTFOLIO_DATA_2")
    if env: paths.append(Path(env).expanduser())
    root = Path(__file__).resolve().parents[2]
    paths.extend([root / "data" / "portfolio_data_2.npz", root / "portfolio_data_2.npz", Path.cwd() / "data" / "portfolio_data_2.npz", Path.cwd() / "portfolio_data_2.npz"])
    return paths


def locate_sector_dataset() -> Path:
    for path in _candidate_paths():
        if path.exists(): return path
    raise FileNotFoundError("Multi-sector validation data not found. Run `python src/fetch_sector_data.py` once, or set VANGUARD_PORTFOLIO_DATA_2 to the supplied portfolio_data_2.npz path.")


def _load_prefix(archive: np.lib.npyio.NpzFile, prefix: str) -> SourceMomentData:
    return SourceMomentData(tickers=tuple(str(x) for x in archive[f"{prefix}_tickers"].tolist()), latest_prices=np.asarray(archive[f"{prefix}_latest_prices"], dtype=float), expected_returns=np.asarray(archive[f"{prefix}_exp_returns"], dtype=float), covariance=np.asarray(archive[f"{prefix}_cov_matrix"], dtype=float), co_skewness=np.asarray(archive[f"{prefix}_co_skewness"], dtype=float), co_kurtosis=np.asarray(archive[f"{prefix}_co_kurtosis"], dtype=float))


def load_sector_dataset_bundle(path: str | Path | None = None) -> Dict[str, object]:
    resolved = Path(path).expanduser() if path is not None else locate_sector_dataset()
    with np.load(resolved, allow_pickle=False) as archive:
        train = _load_prefix(archive, "train"); sectors = [_load_prefix(archive, f"test_{i}") for i in range(10)]
    return {"path": str(resolved), "train": train, "validation": sectors[:5], "held_out": sectors[5:], "validation_names": list(VALIDATION_NAMES), "held_out_names": list(HELD_OUT_NAMES), "split_note": "Generation/model selection uses only test_0..test_4. test_5..test_9 are evaluated only after G* is fixed."}
