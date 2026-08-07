"""Vanguard-aligned multi-asset portfolio co-pilot package."""

from .model import PROFILES, InvestorProfile, synthetic_asset_class_data
from .workflow import challenge_report, run_challenge

__all__ = [
    "PROFILES",
    "InvestorProfile",
    "challenge_report",
    "run_challenge",
    "synthetic_asset_class_data",
]
