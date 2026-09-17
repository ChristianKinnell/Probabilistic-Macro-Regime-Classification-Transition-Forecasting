"""End-to-end smoke test.

Runs the full pipeline (main()) with use_live_data=False, which forces the
regime-aware synthetic-data generator -- the same offline path the module
docstring describes as existing specifically so the pipeline is testable
without network access. Monte Carlo / bootstrap draw counts are reduced
so this runs in well under a minute; it is not a check on the *quality*
of the model's output, only that every stage completes and hands back a
structurally sane result. Default hyperparameters (z_window,
min_regime_history, refit_every) are left at their Config() defaults and
only the start_date/end_date are left at defaults too (2007-06-01 ->
today) -- the pipeline needs on the order of 6-8 years of warmup before
its first classified month (feature z-scoring + PCA gauge warmup +
another min_regime_history before the first GMM/HMM refit), so a short
custom window will silently classify zero months. See docs/LIMITATIONS.md
for the synthetic-data-mode caveats this test does NOT check for (e.g.
that synthetic series share latent factors and so don't validate
cross-series independence -- this test only validates that the machinery
runs).
"""
import warnings

import pytest

from macro_regime_classifier import Config, main


@pytest.fixture(scope="module")
def pipeline_results():
    warnings.filterwarnings("ignore")
    cfg = Config(
        use_live_data=False,
        hmm_n_restarts=1,
        hmm_n_iter=50,
        n_bootstrap=20,
        n_permutation=20,
        confidence_n_bootstrap=10,
        confidence_bootstrap_hmm_n_iter=20,
        validation_state_stability_n_boot=5,
        forecast_mc_paths=200,
    )
    return main(cfg)


def test_pipeline_runs_and_returns_expected_keys(pipeline_results):
    expected_keys = {
        "macro", "prices", "features", "regime_panel", "quad_fit", "vol_liq_fit",
        "validator", "alloc", "quadrant_risk_table", "pit_risk", "suite",
        "gauge_info", "truth_df", "forecast", "confidence", "validation", "cross_asset",
    }
    assert expected_keys.issubset(pipeline_results.keys())


def test_regime_panel_has_both_axes_and_no_gaps(pipeline_results):
    panel = pipeline_results["regime_panel"]
    assert len(panel) > 0, "must classify at least some months over the default ~19yr window"
    assert "quadrant" in panel.columns
    assert "vol_liquidity" in panel.columns
    assert panel["quadrant"].isna().sum() == 0
    assert panel["vol_liquidity"].isna().sum() == 0

    valid_quadrants = {"Stable Growth", "Overheating", "Stagflation", "Contraction / Disinflation"}
    valid_vol_liq = {"Calm", "Stress", "Crisis"}
    assert set(panel["quadrant"].unique()).issubset(valid_quadrants)
    assert set(panel["vol_liquidity"].unique()).issubset(valid_vol_liq)


def test_posterior_probabilities_sum_to_one(pipeline_results):
    panel = pipeline_results["regime_panel"]
    quad_cols = [c for c in panel.columns if c.startswith("quadrant_proba_")]
    vol_cols = [c for c in panel.columns if c.startswith("vol_liquidity_proba_")]
    assert len(quad_cols) == 4 and len(vol_cols) == 3

    assert (panel[quad_cols].sum(axis=1) - 1.0).abs().max() < 1e-6
    assert (panel[vol_cols].sum(axis=1) - 1.0).abs().max() < 1e-6


def test_synthetic_ground_truth_check_ran(pipeline_results):
    """use_live_data=False must trigger the synthetic-data path, which is
    the only path that produces a truth_df for the recovery-accuracy
    sanity check described in the module docstring."""
    assert pipeline_results["truth_df"] is not None
    assert "true_quadrant" in pipeline_results["truth_df"].columns


def test_defensibility_suite_and_validation_populated(pipeline_results):
    suite = pipeline_results["suite"]
    for key in ("bootstrap_sharpe_ci_strategy", "regime_permutation_test", "holm_bonferroni"):
        assert key in suite

    validation = pipeline_results["validation"]
    assert set(validation.keys()) == {"A", "B", "C", "D"}
