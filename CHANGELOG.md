# Changelog

## 0.2.0

- Renamed `Macro Regime Classifier.py` → `macro_regime_classifier.py` (git history preserved via rename) and added `run.py` as the documented entry point.
- Added `requirements.txt` — the README's install instructions are now actually reproducible (previously referenced a `pyproject.toml`/`pip install -e .` that didn't exist in the repo).
- Moved `METHODOLOGY.md`, `VALIDATION.md`, `LIMITATIONS.md` into `docs/`, matching the links the README already pointed at.
- Added `tests/` (fast unit tests + an offline end-to-end smoke test against the synthetic-data fallback path) and a GitHub Actions workflow running them on every push.
- Rewrote the module docstring to accurately describe what the ten pipeline stages do and how they relate to `docs/METHODOLOGY.md`'s design spec — the previous version mischaracterized the tactical allocation as a co-equal downstream product rather than the demonstration layer the code and spec both already treat it as.
- Removed `GITHUB_SETUP.md` (one-time setup instructions for a repo that already exists) and added an explicit "License: not yet chosen" note to the README instead of a silent gap.
- Rewrote README's Installation/Run/Repository-structure sections to match what's actually in the repo.

## 0.1.0

- Two-axis economic and financial-stress regime architecture.
- Point-in-time FRED/ALFRED data layer with vintage-aware YoY feature reconstruction.
- Walk-forward PCA feature factors with Level / Direction / Acceleration diagnostics.
- GMM benchmark plus causally filtered HMM primary state estimator.
- Probability-based regime confirmation and transition-watch logic.
- Multi-horizon transition and recession forecasting.
- Confidence, entropy, OOD, bootstrap and stability diagnostics.
- Dedicated classifier validation and economic reality-check suite.
- Cross-asset regime mapping and demonstration tactical allocation.
