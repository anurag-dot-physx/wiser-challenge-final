# Data provenance

The **flagship eight-asset portfolio model does not depend on proprietary Vanguard data**. Its return, covariance, income, transaction-cost, current-allocation, and scenario assumptions are deterministic synthetic/anonymized inputs defined in `src/vanguard_copilot/model.py`.

The `source_portfolio_snapshot.json` file is used only by the exploratory higher-moment appendix. It contains public-market-derived moments for AAPL, MSFT, TSLA, and AMZN over the source experiment's historical window. It is not used to make the flagship production recommendation.

For the optional validation-selected higher-moment workflow, generate the 1-train + 10-sector public-market dataset with:

```bash
python src/fetch_sector_data.py
```

This creates `data/portfolio_data_2.npz`. The split is fixed before model selection:

- `test_0`–`test_4`: validation/model-selection sectors;
- `test_5`–`test_9`: held-out sectors evaluated only after `G*` and the HUBO coefficients are fixed.

The generated NPZ is intentionally not required for the flagship co-pilot, audit, or test suite. This keeps the principal challenge solution reproducible without network access and separates the public-market research appendix from the synthetic/anonymized challenge solution.
