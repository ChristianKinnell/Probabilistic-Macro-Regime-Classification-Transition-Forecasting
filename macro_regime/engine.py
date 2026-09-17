from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from math import log

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import PointInTimeDataset


@dataclass(frozen=True)
class AxisConfig:
    name: str
    features: tuple[str, str]
    n_regimes: int = 4
    n_components: int = 2
    random_state: int = 0


class AxisModel:
    def __init__(self, config: AxisConfig) -> None:
        self.config = config
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.gmm: GaussianMixture | None = None
        self.hmm: CategoricalHMM | None = None
        self.empirical_transition: np.ndarray | None = None
        self.state_labels: dict[int, str] = {}
        self.state_feature_levels: dict[int, tuple[str, str]] = {}
        self.feature_thresholds: tuple[float, float] | None = None

    def fit(self, dataset: PointInTimeDataset) -> None:
        frame = dataset.time_series()
        if len(frame) < 2:
            raise ValueError(f"{self.config.name} requires at least two observations.")

        features = frame.loc[:, list(self.config.features)].astype(float)
        n_components = min(self.config.n_components, len(self.config.features), len(features))
        n_regimes = max(1, min(self.config.n_regimes, len(features) // 2 or 1))

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)

        self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
        reduced = self.pca.fit_transform(scaled)

        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=self.config.random_state,
        )
        states = self.gmm.fit_predict(reduced)
        self.feature_thresholds = tuple(features.median().tolist())  # type: ignore[assignment]
        self.state_labels, self.state_feature_levels = self._build_state_labels(features, states, n_regimes)
        self.empirical_transition = self._build_empirical_transition(states, n_regimes)
        self.hmm = self._fit_hmm(states, n_regimes)

    def classify(self, dataset: PointInTimeDataset, as_of: date) -> dict[str, object]:
        self._ensure_fit()
        frame = dataset.time_series(as_of=as_of)
        if frame.empty:
            raise ValueError(f"No point-in-time observations available for {as_of.isoformat()}.")

        features = frame.loc[:, list(self.config.features)].astype(float)
        reduced = self.pca.transform(self.scaler.transform(features))
        current_probabilities = self.gmm.predict_proba(reduced)[-1]
        current_state = int(np.argmax(current_probabilities))

        states = self.gmm.predict(reduced)
        transition_probabilities = self._forecast_next_state_probabilities(states, current_probabilities)
        return {
            "axis": self.config.name,
            "as_of": as_of.isoformat(),
            "state_id": current_state,
            "label": self.state_labels[current_state],
            "probabilities": self._format_probabilities(current_probabilities),
            "confidence": float(np.max(current_probabilities)),
            "uncertainty": float(self._normalized_entropy(current_probabilities)),
            "transition_probabilities": self._format_probabilities(transition_probabilities),
        }

    def in_sample_states(self, dataset: PointInTimeDataset) -> pd.DataFrame:
        self._ensure_fit()
        frame = dataset.time_series()
        reduced = self.pca.transform(self.scaler.transform(frame.loc[:, list(self.config.features)].astype(float)))
        states = self.gmm.predict(reduced)
        result = frame.loc[:, ["observed_at"]].copy()
        result["state_id"] = states
        result["label"] = [self.state_labels[int(state)] for state in states]
        return result

    def _fit_hmm(self, states: np.ndarray, n_regimes: int) -> CategoricalHMM | None:
        minimum_observations = max(3, n_regimes * n_regimes * 2)
        if len(states) < minimum_observations or n_regimes < 2:
            return None
        hmm = CategoricalHMM(
            n_components=n_regimes,
            n_features=n_regimes,
            n_iter=200,
            params="st",
            init_params="st",
            random_state=self.config.random_state,
        )
        try:
            hmm.startprob_ = np.full(n_regimes, 1.0 / n_regimes)
            hmm.transmat_ = self.empirical_transition.copy()
            hmm.emissionprob_ = np.eye(n_regimes)
            hmm.fit(states.reshape(-1, 1))
            return hmm
        except Exception:
            return None

    @staticmethod
    def _build_empirical_transition(states: np.ndarray, n_regimes: int) -> np.ndarray:
        counts = np.ones((n_regimes, n_regimes))
        for current_state, next_state in zip(states[:-1], states[1:], strict=False):
            counts[int(current_state), int(next_state)] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def _build_state_labels(
        self,
        features: pd.DataFrame,
        states: np.ndarray,
        n_regimes: int,
    ) -> tuple[dict[int, str], dict[int, tuple[str, str]]]:
        first_feature, second_feature = self.config.features
        first_name = first_feature.replace("_", " ").title()
        second_name = second_feature.replace("_", " ").title()
        first_threshold, second_threshold = self.feature_thresholds or (0.0, 0.0)
        labels: dict[int, str] = {}
        feature_levels: dict[int, tuple[str, str]] = {}
        used: set[str] = set()
        for state_id in range(n_regimes):
            cluster = features.iloc[states == state_id]
            center = cluster.mean() if not cluster.empty else features.mean()
            first_level = "High" if float(center.iloc[0]) >= first_threshold else "Low"
            second_level = "High" if float(center.iloc[1]) >= second_threshold else "Low"
            feature_levels[state_id] = (first_level, second_level)
            label = f"{first_name} {first_level} / {second_name} {second_level}"
            unique_label = label
            suffix = 2
            while unique_label in used:
                unique_label = f"{label} ({suffix})"
                suffix += 1
            used.add(unique_label)
            labels[state_id] = unique_label
        return labels, feature_levels

    def _forecast_next_state_probabilities(self, states: np.ndarray, current_probabilities: np.ndarray) -> np.ndarray:
        if self.hmm is not None:
            next_probabilities = self.hmm.predict_proba(states.reshape(-1, 1))[-1] @ self.hmm.transmat_
            total = float(next_probabilities.sum())
            if total > 0:
                return next_probabilities / total
        return current_probabilities @ self.empirical_transition

    def _format_probabilities(self, probabilities: Iterable[float]) -> dict[str, float]:
        return {
            self.state_labels[state_id]: float(probability)
            for state_id, probability in enumerate(probabilities)
        }

    @staticmethod
    def _normalized_entropy(probabilities: np.ndarray) -> float:
        safe = probabilities[probabilities > 0]
        if len(probabilities) <= 1 or safe.size == 0:
            return 0.0
        entropy = -sum(float(p) * log(float(p)) for p in safe)
        max_entropy = log(len(probabilities))
        return entropy / max_entropy if max_entropy else 0.0

    def _ensure_fit(self) -> None:
        if not all(
            [
                self.scaler,
                self.pca,
                self.gmm,
                self.state_labels,
                self.state_feature_levels,
                self.empirical_transition is not None,
            ]
        ):
            raise RuntimeError(f"{self.config.name} has not been fit yet.")


class TwoAxisMacroRegimeEngine:
    def __init__(self, growth_inflation: AxisConfig, volatility_liquidity: AxisConfig) -> None:
        self.growth_inflation_model = AxisModel(growth_inflation)
        self.volatility_liquidity_model = AxisModel(volatility_liquidity)

    def fit(self, dataset: PointInTimeDataset) -> "TwoAxisMacroRegimeEngine":
        self.growth_inflation_model.fit(dataset)
        self.volatility_liquidity_model.fit(dataset)
        return self

    def classify(self, dataset: PointInTimeDataset, as_of: date) -> dict[str, object]:
        growth_state = self.growth_inflation_model.classify(dataset, as_of)
        volatility_state = self.volatility_liquidity_model.classify(dataset, as_of)
        confidence = float((growth_state["confidence"] + volatility_state["confidence"]) / 2.0)
        uncertainty = float((growth_state["uncertainty"] + volatility_state["uncertainty"]) / 2.0)
        return {
            "as_of": as_of.isoformat(),
            "joint_state": f"{growth_state['label']} × {volatility_state['label']}",
            "confidence": confidence,
            "uncertainty": uncertainty,
            "axes": {
                "growth_inflation": growth_state,
                "volatility_liquidity": volatility_state,
            },
        }

    def validate_against_recessions(
        self,
        dataset: PointInTimeDataset,
        recession_periods: list[tuple[date, date]],
    ) -> dict[str, float]:
        states = self.growth_inflation_model.in_sample_states(dataset)
        if states.empty:
            return {"precision": 0.0, "recall": 0.0, "accuracy": 0.0}

        flags = states["observed_at"].dt.date.apply(lambda day: self._is_recession(day, recession_periods))
        predicted_state_ids = {
            state_id
            for state_id, (first_level, _) in self.growth_inflation_model.state_feature_levels.items()
            if first_level == "Low"
        }
        predicted = states["state_id"].isin(predicted_state_ids)

        tp = int((predicted & flags).sum())
        fp = int((predicted & ~flags).sum())
        fn = int((~predicted & flags).sum())
        tn = int((~predicted & ~flags).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        return {"precision": precision, "recall": recall, "accuracy": accuracy}

    def asset_regime_map(self, dataset: PointInTimeDataset) -> dict[str, dict[str, float]]:
        frame = dataset.time_series()
        if frame.empty or not dataset.asset_names:
            return {}

        growth_states = self.growth_inflation_model.in_sample_states(dataset).rename(
            columns={"state_id": "growth_state_id", "label": "growth_label"}
        )
        volatility_states = self.volatility_liquidity_model.in_sample_states(dataset).rename(
            columns={"state_id": "volatility_state_id", "label": "volatility_label"}
        )
        merged = (
            frame.merge(growth_states, on="observed_at", how="inner")
            .merge(volatility_states, on="observed_at", how="inner")
            .sort_values("observed_at")
        )
        grouped_returns: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for _, row in merged.iterrows():
            label = f"{row['growth_label']} × {row['volatility_label']}"
            for asset in dataset.asset_names:
                column = f"asset::{asset}"
                if column in row and not pd.isna(row[column]):
                    grouped_returns[label][asset].append(float(row[column]))

        return {
            regime: {
                asset: float(np.mean(values))
                for asset, values in asset_groups.items()
                if values
            }
            for regime, asset_groups in grouped_returns.items()
        }

    @staticmethod
    def _is_recession(day: date, recession_periods: list[tuple[date, date]]) -> bool:
        return any(start <= day <= end for start, end in recession_periods)
