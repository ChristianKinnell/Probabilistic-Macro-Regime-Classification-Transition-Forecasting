"""Fast unit tests for standalone pieces of the classifier that don't require
running the full walk-forward pipeline. See test_smoke.py for an end-to-end
run.
"""
import numpy as np
import pandas as pd
import pytest

from macro_regime_classifier import (
    Config,
    assign_states_to_prototypes,
    QUADRANT_PROTOTYPES,
    holm_bonferroni,
    rolling_zscore,
    _regime_segments,
    expected_state_duration_analytic,
    confirm_regime_sequence,
)


def test_assign_states_to_prototypes_recovers_exact_match():
    """Four cluster means placed exactly at the four canonical prototypes
    must map back to those same labels with zero distance, regardless of
    the order the clusters come out of the GMM in."""
    labels = list(QUADRANT_PROTOTYPES.keys())
    # Deliberately shuffled/rotated order vs. QUADRANT_PROTOTYPES' own order.
    means = np.array([QUADRANT_PROTOTYPES[lab] for lab in reversed(labels)])
    label_map, distances = assign_states_to_prototypes(means, QUADRANT_PROTOTYPES)

    assert set(label_map.values()) == set(labels), "must be a one-to-one assignment"
    for lab in labels:
        assert distances[lab] == pytest.approx(0.0, abs=1e-9)


def test_assign_states_to_prototypes_is_one_to_one():
    """Two cluster means close to the *same* prototype must still be split
    across two distinct labels, not collapsed onto one -- this is exactly
    the failure mode Hungarian assignment exists to prevent."""
    means = np.array([
        [0.9, -0.9],   # near Stable Growth
        [1.1, -1.1],   # also near Stable Growth
        [-1.0, 1.0],   # near Stagflation
        [-1.0, -1.0],  # near Contraction
    ])
    label_map, _ = assign_states_to_prototypes(means, QUADRANT_PROTOTYPES)
    assert len(set(label_map.values())) == 4, "every state must get a distinct label"


def test_holm_bonferroni_monotonic_and_conservative():
    pvals = {"a": 0.001, "b": 0.01, "c": 0.20, "d": 0.50}
    adj = holm_bonferroni(pvals, alpha=0.05)

    for name, p_raw in pvals.items():
        assert adj[name]["p_holm"] >= p_raw - 1e-12, "Holm p-values never shrink the raw p-value"
    assert adj["a"]["significant_at_alpha"] is True
    assert adj["d"]["significant_at_alpha"] is False


def test_rolling_zscore_clips_and_centers():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1e-6, 30)  # near-flat but non-degenerate (std != 0)
    s = pd.Series(np.concatenate([base, [1000.0]]))  # one extreme outlier
    z = rolling_zscore(s, window=12, clip=5.0)
    assert z.dropna().abs().max() <= 5.0 + 1e-9, "clip bound must be respected"
    # A flat series (excluding the outlier) should have a small z-score just before it.
    assert abs(z.iloc[29]) < 1.0

    # A genuinely constant window (zero variance) must not raise -- it should
    # come out as NaN via the function's own divide-by-zero guard.
    flat = pd.Series(np.zeros(20))
    z_flat = rolling_zscore(flat, window=12, clip=5.0)
    assert z_flat.iloc[15:].isna().all()


def test_regime_segments_splits_on_label_change():
    labels = pd.Series(
        ["A", "A", "A", "B", "B", "A", "A"],
        index=pd.date_range("2020-01-01", periods=7, freq="MS"),
    )
    segs = _regime_segments(labels)
    seg_labels = [lab for _, _, lab in segs]
    assert seg_labels == ["A", "B", "A"]
    # first segment covers exactly the first three months
    assert segs[0][0] == labels.index[0]
    assert segs[0][1] == labels.index[2]


def test_expected_state_duration_analytic_matches_geometric_formula():
    labels = ["Calm", "Stress", "Crisis"]
    transmat = np.array([
        [0.90, 0.08, 0.02],
        [0.20, 0.70, 0.10],
        [0.30, 0.30, 0.40],
    ])
    dur = expected_state_duration_analytic(transmat, labels)
    assert dur["Calm"] == pytest.approx(1.0 / (1 - 0.90))
    assert dur["Crisis"] == pytest.approx(1.0 / (1 - 0.40))


def test_confirm_regime_sequence_requires_persistence_before_switching():
    """A single high-confidence off-incumbent reading should NOT immediately
    flip the confirmed regime unless it clears the override threshold -- it
    should register as 'Transition Watch' until it persists for n_confirm
    periods."""
    cfg = Config()
    labels = ["Calm", "Stress", "Crisis"]
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")

    # Stays confidently in Calm, then one period leans Stress at a level
    # that clears p_enter/margin but sits below the override threshold,
    # then reverts to Calm.
    rows = [
        {"Calm": 0.90, "Stress": 0.08, "Crisis": 0.02},
        {"Calm": 0.90, "Stress": 0.08, "Crisis": 0.02},
        {"Calm": 0.30, "Stress": 0.65, "Crisis": 0.05},   # single off-incumbent reading
        {"Calm": 0.90, "Stress": 0.08, "Crisis": 0.02},
        {"Calm": 0.90, "Stress": 0.08, "Crisis": 0.02},
    ]
    posteriors = pd.DataFrame(rows, index=idx)[labels]
    gmm_labels = posteriors.idxmax(axis=1)
    transmat = np.array([
        [0.90, 0.08, 0.02],
        [0.20, 0.70, 0.10],
        [0.30, 0.30, 0.40],
    ])
    active_transmat = pd.Series([transmat] * len(idx), index=idx)

    out = confirm_regime_sequence(posteriors, gmm_labels, active_transmat, labels, cfg, axis="stress")

    assert out["confirmed_regime"].iloc[2] == "Calm", (
        "a single off-incumbent reading below the override threshold must not "
        "flip the confirmed regime on its own"
    )
    assert out.loc[idx[2], "status"] in ("Transition Watch", "Confirmed")
