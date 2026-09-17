from __future__ import annotations

import unittest
from datetime import date

from macro_regime import AxisConfig, PointInTimeDataset, PointInTimeRecord, TwoAxisMacroRegimeEngine


def build_dataset() -> PointInTimeDataset:
    return build_dataset_with_months(24)


def build_dataset_with_months(months: int) -> PointInTimeDataset:
    records: list[PointInTimeRecord] = []
    observed_dates = [date(2020 + (month - 1) // 12, ((month - 1) % 12) + 1, 1) for month in range(1, months + 1)]
    regimes = [
        (2.0, -1.5, -1.0, 1.5, 0.030, -0.005),
        (1.8, 1.7, -0.8, -1.2, 0.020, -0.015),
        (-1.9, 1.8, 1.9, -1.5, -0.035, 0.020),
        (-1.7, -1.4, 1.6, 1.2, -0.010, 0.015),
    ]
    for index, observed_at in enumerate(observed_dates):
        growth, inflation, volatility, liquidity, equities, bonds = regimes[index % len(regimes)]
        base_values = {
            "growth": growth + 0.05 * (index % 3),
            "inflation": inflation - 0.03 * (index % 2),
            "volatility": volatility + 0.04 * (index % 2),
            "liquidity": liquidity - 0.02 * (index % 3),
        }
        records.append(
            PointInTimeRecord(
                observed_at=observed_at,
                available_at=observed_at,
                values=base_values,
                asset_returns={"equities": equities, "bonds": bonds},
            )
        )
        if index < min(6, months):
            revised_month = observed_at.month + 1
            revised_year = observed_at.year + (1 if revised_month == 13 else 0)
            revised_month = 1 if revised_month == 13 else revised_month
            records.append(
                PointInTimeRecord(
                    observed_at=observed_at,
                    available_at=date(revised_year, revised_month, 1),
                    values={**base_values, "growth": base_values["growth"] + 0.4},
                    asset_returns={"equities": equities, "bonds": bonds},
                )
            )
    return PointInTimeDataset(records)


class MacroRegimeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = build_dataset()
        self.engine = TwoAxisMacroRegimeEngine(
            growth_inflation=AxisConfig(name="growth_inflation", features=("growth", "inflation")),
            volatility_liquidity=AxisConfig(name="volatility_liquidity", features=("volatility", "liquidity")),
        ).fit(self.dataset)

    def test_point_in_time_snapshot_uses_latest_available_release(self) -> None:
        january = self.dataset.snapshot(date(2020, 1, 1))
        february = self.dataset.snapshot(date(2020, 2, 1))
        january_growth_initial = float(january.loc[january["observed_at"].dt.date == date(2020, 1, 1), "growth"].iloc[0])
        january_growth_revised = float(february.loc[february["observed_at"].dt.date == date(2020, 1, 1), "growth"].iloc[0])
        self.assertLess(january_growth_initial, january_growth_revised)

    def test_point_in_time_series_respects_requested_cutoff(self) -> None:
        january_series = self.dataset.time_series(as_of=date(2020, 1, 1))
        february_series = self.dataset.time_series(as_of=date(2020, 2, 1))
        january_growth_initial = float(january_series.loc[january_series["observed_at"].dt.date == date(2020, 1, 1), "growth"].iloc[0])
        january_growth_revised = float(february_series.loc[february_series["observed_at"].dt.date == date(2020, 1, 1), "growth"].iloc[0])
        self.assertLess(january_growth_initial, january_growth_revised)

    def test_classification_returns_joint_state_and_probabilities(self) -> None:
        result = self.engine.classify(self.dataset, date(2021, 12, 1))
        self.assertIn("joint_state", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertGreaterEqual(result["uncertainty"], 0.0)
        self.assertLessEqual(result["uncertainty"], 1.0)
        growth_probs = result["axes"]["growth_inflation"]["transition_probabilities"]
        self.assertAlmostEqual(sum(growth_probs.values()), 1.0, places=6)

    def test_small_samples_use_empirical_transition_fallback(self) -> None:
        small_dataset = build_dataset_with_months(6)
        small_engine = TwoAxisMacroRegimeEngine(
            growth_inflation=AxisConfig(name="growth_inflation", features=("growth", "inflation")),
            volatility_liquidity=AxisConfig(name="volatility_liquidity", features=("volatility", "liquidity")),
        ).fit(small_dataset)
        self.assertIsNone(small_engine.growth_inflation_model.hmm)
        result = small_engine.classify(small_dataset, date(2020, 6, 1))
        self.assertAlmostEqual(
            sum(result["axes"]["growth_inflation"]["transition_probabilities"].values()),
            1.0,
            places=6,
        )

    def test_recession_validation_returns_metrics(self) -> None:
        metrics = self.engine.validate_against_recessions(
            self.dataset,
            recession_periods=[(date(2020, 3, 1), date(2020, 8, 1)), (date(2021, 7, 1), date(2021, 10, 1))],
        )
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_asset_regime_map_groups_asset_behaviour_by_regime(self) -> None:
        mapping = self.engine.asset_regime_map(self.dataset)
        self.assertTrue(mapping)
        any_regime = next(iter(mapping.values()))
        self.assertIn("equities", any_regime)
        self.assertIn("bonds", any_regime)


if __name__ == "__main__":
    unittest.main()
