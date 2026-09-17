# Probabilistic-Macro-Regime-Classification-Transition-Forecasting

Systematic two-axis macro regime engine using point-in-time data, PCA, GMM and HMM models to classify Growth × Inflation and Volatility × Liquidity states, forecast regime transitions, quantify confidence and uncertainty, validate signals against NBER recessions, and map regime-dependent behaviour across global asset classes.

## What is included

- Point-in-time dataset support with revision-aware snapshots
- PCA + Gaussian mixture state classification for each macro axis
- HMM-based transition probability forecasts over classified states
- Confidence and uncertainty estimates from regime posterior probabilities
- Recession alignment metrics for the Growth × Inflation axis
- Regime-conditioned asset return mapping

## Package layout

- `macro_regime/data.py` contains point-in-time data structures
- `macro_regime/engine.py` contains the axis models and two-axis engine
- `tests/test_engine.py` contains end-to-end unit coverage on synthetic data

## Quick start

```python
from datetime import date

from macro_regime import AxisConfig, PointInTimeDataset, PointInTimeRecord, TwoAxisMacroRegimeEngine

dataset = PointInTimeDataset(
    [
        PointInTimeRecord(
            observed_at=date(2024, 1, 1),
            available_at=date(2024, 1, 1),
            values={
                "growth": 1.2,
                "inflation": 2.1,
                "volatility": -0.3,
                "liquidity": 0.8,
            },
            asset_returns={"equities": 0.015, "bonds": 0.004},
        ),
        PointInTimeRecord(
            observed_at=date(2024, 2, 1),
            available_at=date(2024, 2, 1),
            values={
                "growth": 0.9,
                "inflation": 2.4,
                "volatility": 0.1,
                "liquidity": 0.5,
            },
            asset_returns={"equities": 0.008, "bonds": 0.002},
        ),
        PointInTimeRecord(
            observed_at=date(2024, 3, 1),
            available_at=date(2024, 3, 1),
            values={
                "growth": -0.7,
                "inflation": 3.0,
                "volatility": 0.9,
                "liquidity": -0.4,
            },
            asset_returns={"equities": -0.011, "bonds": 0.007},
        ),
        PointInTimeRecord(
            observed_at=date(2024, 4, 1),
            available_at=date(2024, 4, 1),
            values={
                "growth": -1.0,
                "inflation": 1.8,
                "volatility": 1.1,
                "liquidity": -0.6,
            },
            asset_returns={"equities": -0.006, "bonds": 0.009},
        ),
    ]
)

engine = TwoAxisMacroRegimeEngine(
    growth_inflation=AxisConfig(
        name="growth_inflation",
        features=("growth", "inflation"),
        n_regimes=2,
    ),
    volatility_liquidity=AxisConfig(
        name="volatility_liquidity",
        features=("volatility", "liquidity"),
        n_regimes=2,
    ),
).fit(dataset)

classification = engine.classify(dataset, as_of=date(2024, 2, 1))
recession_metrics = engine.validate_against_recessions(
    dataset,
    recession_periods=[(date(2024, 1, 1), date(2024, 1, 31))],
)
asset_map = engine.asset_regime_map(dataset)
```

## Running tests

```bash
python -m unittest discover -s tests
```
