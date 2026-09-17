import unittest

from macro_regime_engine import MacroObservation, MacroRegimeEngine


class MacroRegimeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MacroRegimeEngine(
            growth_features=["growth"],
            inflation_features=["inflation"],
            volatility_features=["volatility"],
            liquidity_features=["liquidity"],
        )

    def test_end_to_end_report_contains_probabilities_forecasts_and_mappings(self) -> None:
        observations = [
            MacroObservation(
                timestamp=f"2021-0{month}-28",
                available_at=f"2021-0{month}-28",
                macro_features=macro_features,
                asset_returns=asset_returns,
                nber_recession=nber_recession,
            )
            for month, macro_features, asset_returns, nber_recession in [
                (1, {"growth": 1.6, "inflation": 1.2, "volatility": -0.8, "liquidity": 1.0}, {"GlobalEquities": 0.03, "Bonds": -0.01}, False),
                (2, {"growth": 1.1, "inflation": 0.8, "volatility": -0.6, "liquidity": 0.9}, {"GlobalEquities": 0.02, "Bonds": -0.005}, False),
                (3, {"growth": -1.2, "inflation": 0.7, "volatility": 1.1, "liquidity": -0.8}, {"GlobalEquities": -0.04, "Bonds": 0.015}, True),
                (4, {"growth": -1.4, "inflation": 1.0, "volatility": 1.3, "liquidity": -1.1}, {"GlobalEquities": -0.05, "Bonds": 0.02}, True),
                (5, {"growth": 0.9, "inflation": -1.0, "volatility": -1.2, "liquidity": 1.2}, {"GlobalEquities": 0.025, "Bonds": -0.01}, False),
                (6, {"growth": 1.3, "inflation": -1.1, "volatility": -1.0, "liquidity": 1.4}, {"GlobalEquities": 0.028, "Bonds": -0.008}, False),
                (7, {"growth": -1.0, "inflation": -0.9, "volatility": 0.9, "liquidity": -1.2}, {"GlobalEquities": -0.03, "Bonds": 0.012}, True),
                (8, {"growth": -0.7, "inflation": -1.2, "volatility": 1.2, "liquidity": -1.3}, {"GlobalEquities": -0.02, "Bonds": 0.01}, True),
            ]
        ]

        report = self.engine.fit(observations)

        self.assertEqual(len(report.observations), len(observations))
        self.assertEqual(set(report.latest_forecasts), {"growth_inflation", "volatility_liquidity"})
        self.assertEqual(set(report.transition_matrices), {"growth_inflation", "volatility_liquidity"})
        self.assertGreaterEqual(report.nber_validation["evaluated_observations"], 1.0)
        self.assertGreaterEqual(report.nber_validation["accuracy"], 0.0)
        self.assertLessEqual(report.nber_validation["accuracy"], 1.0)
        self.assertTrue(report.asset_class_behaviour)

        asset_counts = 0.0
        for observation in report.observations:
            for axis in (observation.growth_inflation, observation.volatility_liquidity):
                self.assertAlmostEqual(sum(axis.state_probabilities.values()), 1.0, places=6)
                self.assertAlmostEqual(sum(axis.next_state_probabilities.values()), 1.0, places=6)
                self.assertGreaterEqual(axis.confidence, 0.0)
                self.assertLessEqual(axis.confidence, 1.0)
                self.assertGreaterEqual(axis.uncertainty, 0.0)
                self.assertLessEqual(axis.uncertainty, 1.0)

        for asset_summary in report.asset_class_behaviour.values():
            self.assertIn("GlobalEquities", asset_summary)
            asset_counts += asset_summary["GlobalEquities"]["count"]

        self.assertEqual(asset_counts, float(len(observations)))

    def test_point_in_time_validation_rejects_future_available_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "point-in-time safe"):
            self.engine.fit(
                [
                    MacroObservation(
                        timestamp="2021-01-31",
                        available_at="2021-02-01",
                        macro_features={
                            "growth": 1.0,
                            "inflation": 1.0,
                            "volatility": 1.0,
                            "liquidity": 1.0,
                        },
                    )
                ]
            )

    def test_missing_required_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required macro feature: liquidity"):
            self.engine.fit(
                [
                    MacroObservation(
                        timestamp="2021-01-31",
                        available_at="2021-01-31",
                        macro_features={
                            "growth": 1.0,
                            "inflation": 1.0,
                            "volatility": 1.0,
                        },
                    )
                ]
            )

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one observation is required"):
            self.engine.fit([])

    def test_component_count_is_clamped_to_sample_count(self) -> None:
        engine = MacroRegimeEngine(
            growth_features=["growth"],
            inflation_features=["inflation"],
            volatility_features=["volatility"],
            liquidity_features=["liquidity"],
            n_components=8,
        )
        report = engine.fit(
            [
                MacroObservation(
                    timestamp="2021-01-31",
                    available_at="2021-01-31",
                    macro_features={
                        "growth": 1.0,
                        "inflation": 0.5,
                        "volatility": -0.5,
                        "liquidity": 0.8,
                    },
                ),
                MacroObservation(
                    timestamp="2021-02-28",
                    available_at="2021-02-28",
                    macro_features={
                        "growth": -1.0,
                        "inflation": -0.5,
                        "volatility": 0.6,
                        "liquidity": -0.9,
                    },
                ),
                MacroObservation(
                    timestamp="2021-03-31",
                    available_at="2021-03-31",
                    macro_features={
                        "growth": 0.9,
                        "inflation": -0.7,
                        "volatility": -0.8,
                        "liquidity": 1.0,
                    },
                ),
            ]
        )

        self.assertEqual(len(report.observations[0].growth_inflation.state_probabilities), 3)
        self.assertEqual(len(report.observations[0].volatility_liquidity.state_probabilities), 3)

    def test_empty_feature_groups_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "growth_features must contain at least one feature"):
            MacroRegimeEngine(
                growth_features=[],
                inflation_features=["inflation"],
                volatility_features=["volatility"],
                liquidity_features=["liquidity"],
            )

    def test_nber_precision_and_recall_are_exact(self) -> None:
        report = self.engine.fit(
            [
                MacroObservation(
                    timestamp="2021-01-31",
                    available_at="2021-01-31",
                    macro_features={"growth": -2.0, "inflation": 0.5, "volatility": 1.0, "liquidity": -1.0},
                    nber_recession=True,
                ),
                MacroObservation(
                    timestamp="2021-02-28",
                    available_at="2021-02-28",
                    macro_features={"growth": -1.5, "inflation": 0.4, "volatility": 0.8, "liquidity": -0.8},
                    nber_recession=False,
                ),
                MacroObservation(
                    timestamp="2021-03-31",
                    available_at="2021-03-31",
                    macro_features={"growth": 1.8, "inflation": -0.4, "volatility": -0.8, "liquidity": 0.9},
                    nber_recession=True,
                ),
                MacroObservation(
                    timestamp="2021-04-30",
                    available_at="2021-04-30",
                    macro_features={"growth": 2.1, "inflation": -0.6, "volatility": -1.0, "liquidity": 1.1},
                    nber_recession=False,
                ),
            ]
        )

        self.assertAlmostEqual(report.nber_validation["precision"], 0.5)
        self.assertAlmostEqual(report.nber_validation["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
