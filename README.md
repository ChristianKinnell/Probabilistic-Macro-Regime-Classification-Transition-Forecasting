# Macro Regime Classifier

A point-in-time macro regime research system that estimates **where the economy is, where it may be moving, how confident the model is, and what those regimes imply for cross-asset behaviour**.

The project uses two independent latent-state axes:

- **Growth × Inflation:** Stable Growth, Overheating, Stagflation, Contraction / Disinflation
- **Volatility × Liquidity:** Calm, Stress, Crisis

`Recession` is deliberately modeled separately as an NBER business-cycle event rather than used as a fourth Growth × Inflation quadrant.

## What makes the project different

The model is built as a research pipeline rather than a hard-coded macro playbook. It combines point-in-time macro data, walk-forward feature engineering, GMM/HMM latent-state estimation, probability-based confirmation, transition forecasting, uncertainty diagnostics, dedicated classifier validation, and empirical cross-asset regime mapping.

### Research architecture

1. Regime Definition & Taxonomy
2. Data Universe
3. Point-in-Time Data Engineering
4. Feature Engineering
5. Latent State Estimation
6. Regime Classification & Confirmation
7. Transition & Forecasting Engine
8. Confidence, Uncertainty & Regime Stability
9. Validation & Economic Reality Check
10. Cross-Asset Regime Mapping

The full description of each stage, at implementation-level detail, is in the module docstring at the top of [`macro_regime_classifier.py`](macro_regime_classifier.py). See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the original design-spec version of the same ten stages.

## Statistical engine

The current primary architecture is:

```text
Point-in-time macro data
        ↓
Block-level features + Level / Direction / Acceleration
        ↓
Walk-forward PCA factors
        ↓
GMM benchmark + causal Gaussian HMM
        ↓
Posterior probabilities
        ↓
Probability-based confirmation
        ↓
Transition forecasts + confidence diagnostics
        ↓
Validation + cross-asset mapping
```

Important implementation choices include:

- **Hungarian one-to-one state assignment** to prevent duplicate GMM economic labels.
- **Causal HMM forward filtering** rather than retrospective future-informed decoding.
- **Purged multi-horizon forecast training** to prevent target leakage.
- **ALFRED-aware YoY reconstruction** so numerator and denominator use vintages known at the same decision date.
- **Probability-first reporting** rather than relying on hard regime labels alone.

## Data

Macro data are sourced from FRED/ALFRED. Asset data are sourced from Yahoo Finance with fallback support through `pandas_datareader`/Stooq.

For full vintage-aware reconstruction, provide a FRED API key:

```bash
export FRED_API_KEY="your_key_here"
```

Without a key, the code falls back to current-vintage FRED series with explicit release-lag assumptions. This limitation is reported in the output rather than hidden.

If neither FRED nor Yahoo is reachable at all (e.g. a sandboxed/offline environment), the pipeline falls back further to a regime-aware **synthetic data generator** with a known ground-truth regime path, so it stays fully runnable and testable offline — this is also what the test suite (`tests/test_smoke.py`) runs against.

## Installation

```bash
git clone https://github.com/ChristianKinnell/Probabilistic-Macro-Regime-Classification-Transition-Forecasting..git
cd Probabilistic-Macro-Regime-Classification-Transition-Forecasting.
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

Outputs are written to `outputs/`. The script also builds an interactive Plotly dashboard showing the current regime, posterior probabilities, transition forecasts, uncertainty, recession risk, and cross-asset outlook.

## Tests

```bash
pip install -r requirements.txt   # includes pytest
pytest tests/ -v
```

`tests/test_units.py` covers fast, standalone pieces (Hungarian state assignment, Holm-Bonferroni correction, rolling z-scores, regime-confirmation hysteresis, etc.) with no network or long runtime needed. `tests/test_smoke.py` runs the full pipeline end-to-end offline (`use_live_data=False`, so it exercises the synthetic-data fallback) with reduced bootstrap/Monte-Carlo draw counts, and checks the output is structurally sound (both axes classified with no gaps, posteriors sum to 1, defensibility suite and validation dashboards populated). Neither test checks the *quality* of the model's output — see `docs/LIMITATIONS.md` for what is and isn't validated. CI runs both on every push via `.github/workflows/tests.yml`.

## Validation

The project separates **classifier validation** from **portfolio validation**. Classifier diagnostics include probability calibration, Brier/log-loss metrics, detection lag, regime persistence, bootstrap state stability, model agreement, parameter stability, state-count sensitivity, structural-break analysis, and held-out economic reality checks.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── run.py                        # entry point: `python run.py`
├── macro_regime_classifier.py    # the full research engine (single module, see note below)
├── docs/
│   ├── METHODOLOGY.md
│   ├── VALIDATION.md
│   └── LIMITATIONS.md
├── tests/
│   ├── test_units.py
│   └── test_smoke.py
├── Research Papers for Macro Regime Class/
├── .github/workflows/tests.yml   # CI: installs deps, runs pytest on every push
└── CHANGELOG.md
```

The research engine intentionally stays a single module rather than a `src/`-layout package for now, to minimize refactor risk while the model itself is still actively changing. A later engineering-only refactor can split data, features, regimes, forecasts, validation and reporting into separate modules without changing model behaviour.

## License

Not yet chosen. Treat this as "all rights reserved" until a license file is added — pick one when you decide how permissively you want others to reuse the code (MIT/Apache-2.0 for permissive reuse, or none/proprietary to keep it closed).

## Disclaimer

This repository is a quantitative research project and is not investment advice. Historical regime relationships and simulated/backtested results do not guarantee future performance.
