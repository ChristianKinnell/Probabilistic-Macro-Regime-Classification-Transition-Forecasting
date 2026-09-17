# Probabilistic-Macro-Regime-Classification-Transition-Forecasting
Systematic two-axis macro regime engine using point-in-time data, PCA, GMM and HMM models to classify Growth × Inflation and Volatility × Liquidity states, forecast regime transitions, quantify confidence and uncertainty, validate signals against NBER recessions, and map regime-dependent behaviour across global asset classes.

## Minimal implementation

The repository now includes a small self-contained Python module at `macro_regime_engine.py` that:

- enforces point-in-time safe observations through `available_at <= timestamp`
- compresses feature groups with a lightweight PCA-style first principal component
- fits simple diagonal Gaussian mixtures for Growth × Inflation and Volatility × Liquidity state probabilities
- uses a Markov transition model as an HMM-style filter/forecaster for regime transitions
- reports per-observation confidence, uncertainty, and next-state probabilities
- validates growth-down signals against supplied NBER recession flags
- aggregates asset-class returns by combined regime label

## Running the tests

```bash
python -m unittest discover -s tests -v
```
