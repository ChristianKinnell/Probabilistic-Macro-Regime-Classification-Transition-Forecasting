from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Dict, Iterable, List, Mapping, Sequence


EPSILON = 1e-9


def _parse_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(max(probability, EPSILON))
        for probability in probabilities
        if probability > 0.0
    )
    return entropy / math.log(len(probabilities))


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(column) for column in zip(*matrix)] if matrix else []


def _standardize(matrix: Sequence[Sequence[float]]) -> tuple[List[List[float]], List[float], List[float]]:
    if not matrix:
        return [], [], []

    columns = _transpose(matrix)
    means = [_mean(column) for column in columns]
    stds: List[float] = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / max(len(column), 1)
        stds.append(math.sqrt(variance) or 1.0)

    standardized = []
    for row in matrix:
        standardized.append(
            [(value - mean) / std for value, mean, std in zip(row, means, stds)]
        )
    return standardized, means, stds


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(value * weight for value, weight in zip(row, vector)) for row in matrix]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(value * weight for value, weight in zip(left, right))


def _covariance_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    columns = _transpose(matrix)
    size = len(columns)
    covariance = [[0.0 for _ in range(size)] for _ in range(size)]
    scale = max(len(matrix), 1)
    for row in range(size):
        for column in range(size):
            covariance[row][column] = (
                sum(columns[row][index] * columns[column][index] for index in range(len(matrix)))
                / scale
            )
    return covariance


def _power_iteration(matrix: Sequence[Sequence[float]], iterations: int = 50) -> List[float]:
    size = len(matrix)
    vector = [1.0 / math.sqrt(size)] * size
    for _ in range(iterations):
        candidate = _mat_vec(matrix, vector)
        norm = math.sqrt(sum(value * value for value in candidate)) or 1.0
        vector = [value / norm for value in candidate]
    return vector


def _principal_component_scores(matrix: Sequence[Sequence[float]]) -> List[float]:
    if not matrix:
        return []
    if len(matrix[0]) == 1:
        return [row[0] for row in matrix]
    covariance = _covariance_matrix(matrix)
    component = _power_iteration(covariance)
    if sum(component) < 0.0:
        component = [-value for value in component]
    return [_dot(row, component) for row in matrix]


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _log_sum_exp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


class GaussianMixtureModel:
    def __init__(self, n_components: int = 4, max_iterations: int = 40, tolerance: float = 1e-4) -> None:
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights: List[float] = []
        self.means: List[List[float]] = []
        self.variances: List[List[float]] = []

    def fit(self, points: Sequence[Sequence[float]]) -> "GaussianMixtureModel":
        if not points:
            raise ValueError("cannot fit a Gaussian mixture without observations")

        n_samples = len(points)
        dimension = len(points[0])
        component_count = min(self.n_components, n_samples)
        self.n_components = component_count

        standardized, _, _ = _standardize(points)
        ordering = sorted(range(n_samples), key=lambda index: sum(standardized[index]))
        self.means = []
        for component in range(component_count):
            sample_index = ordering[min(component * n_samples // component_count, n_samples - 1)]
            self.means.append(list(points[sample_index]))

        columns = _transpose(points)
        global_mean = [_mean(column) for column in columns]
        global_variance = []
        for column, mean in zip(columns, global_mean):
            variance = _mean([(value - mean) ** 2 for value in column])
            global_variance.append(max(variance, 1e-4))

        self.variances = [list(global_variance) for _ in range(component_count)]
        self.weights = [1.0 / component_count] * component_count

        previous_log_likelihood: float | None = None
        for _ in range(self.max_iterations):
            responsibilities = self.predict_proba(points)
            counts = [sum(row[component] for row in responsibilities) for component in range(component_count)]

            self.weights = [
                max(count, EPSILON) / max(sum(counts), EPSILON) for count in counts
            ]

            new_means: List[List[float]] = []
            new_variances: List[List[float]] = []
            for component in range(component_count):
                count = max(counts[component], EPSILON)
                mean = []
                variance = []
                for column in range(dimension):
                    weighted_sum = sum(
                        responsibilities[row][component] * points[row][column]
                        for row in range(n_samples)
                    )
                    column_mean = weighted_sum / count
                    mean.append(column_mean)
                    weighted_variance = sum(
                        responsibilities[row][component]
                        * (points[row][column] - column_mean) ** 2
                        for row in range(n_samples)
                    )
                    variance.append(max(weighted_variance / count, 1e-4))
                new_means.append(mean)
                new_variances.append(variance)

            self.means = new_means
            self.variances = new_variances

            log_likelihood = sum(
                _log_sum_exp(self._component_log_probabilities(point)) for point in points
            )
            if previous_log_likelihood is not None and abs(log_likelihood - previous_log_likelihood) < self.tolerance:
                break
            previous_log_likelihood = log_likelihood

        return self

    def _component_log_probabilities(self, point: Sequence[float]) -> List[float]:
        scores = []
        for weight, mean, variance in zip(self.weights, self.means, self.variances):
            normalizer = 0.5 * sum(math.log(2.0 * math.pi * max(value, EPSILON)) for value in variance)
            exponent = 0.5 * sum(
                ((coordinate - center) ** 2) / max(var, EPSILON)
                for coordinate, center, var in zip(point, mean, variance)
            )
            scores.append(math.log(max(weight, EPSILON)) - normalizer - exponent)
        return scores

    def predict_proba(self, points: Sequence[Sequence[float]]) -> List[List[float]]:
        probabilities = []
        for point in points:
            scores = self._component_log_probabilities(point)
            log_total = _log_sum_exp(scores)
            probabilities.append([math.exp(score - log_total) for score in scores])
        return probabilities


class HiddenMarkovModel:
    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.transition_matrix: List[List[float]] = []
        self.initial_probabilities: List[float] = []

    def fit(
        self,
        emissions: Sequence[Sequence[float]],
        decoded_states: Sequence[int],
        prior: Sequence[float],
    ) -> "HiddenMarkovModel":
        if not emissions:
            raise ValueError("cannot fit an HMM without emissions")

        state_count = len(emissions[0])
        aligned_prior = list(prior[:state_count])
        if len(aligned_prior) < state_count:
            aligned_prior.extend([1.0] * (state_count - len(aligned_prior)))
        self.initial_probabilities = _normalize(aligned_prior)
        counts = [
            [self.smoothing for _ in range(state_count)]
            for _ in range(state_count)
        ]
        for current_state, next_state in zip(decoded_states, decoded_states[1:]):
            counts[current_state][next_state] += 1.0
        self.transition_matrix = [_normalize(row) for row in counts]
        return self

    def filter_and_forecast(
        self, emissions: Sequence[Sequence[float]]
    ) -> tuple[List[List[float]], List[List[float]]]:
        filtered_probabilities: List[List[float]] = []
        forecast_probabilities: List[List[float]] = []
        if not emissions:
            return filtered_probabilities, forecast_probabilities

        previous = _normalize(
            [
                self.initial_probabilities[state] * emissions[0][state]
                for state in range(len(self.initial_probabilities))
            ]
        )
        filtered_probabilities.append(previous)
        forecast = [
            sum(previous[source] * self.transition_matrix[source][target] for source in range(len(previous)))
            for target in range(len(previous))
        ]
        forecast_probabilities.append(_normalize(forecast))

        for emission in emissions[1:]:
            predicted = [
                sum(previous[source] * self.transition_matrix[source][target] for source in range(len(previous)))
                for target in range(len(previous))
            ]
            filtered = _normalize(
                [predicted[state] * emission[state] for state in range(len(predicted))]
            )
            forecast = [
                sum(filtered[source] * self.transition_matrix[source][target] for source in range(len(filtered)))
                for target in range(len(filtered))
            ]
            filtered_probabilities.append(filtered)
            forecast_probabilities.append(_normalize(forecast))
            previous = filtered
        return filtered_probabilities, forecast_probabilities


@dataclass(frozen=True)
class MacroObservation:
    timestamp: date | datetime | str
    available_at: date | datetime | str
    macro_features: Mapping[str, float]
    asset_returns: Mapping[str, float] | None = None
    nber_recession: bool | None = None


@dataclass(frozen=True)
class AxisClassification:
    label: str
    state_probabilities: Dict[str, float]
    confidence: float
    uncertainty: float
    next_state_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RegimeObservation:
    timestamp: date
    growth_inflation: AxisClassification
    volatility_liquidity: AxisClassification
    combined_label: str


@dataclass(frozen=True)
class MacroRegimeReport:
    observations: List[RegimeObservation]
    latest_forecasts: Dict[str, Dict[str, float]]
    transition_matrices: Dict[str, Dict[str, Dict[str, float]]]
    nber_validation: Dict[str, float]
    asset_class_behaviour: Dict[str, Dict[str, Dict[str, float]]]


class MacroRegimeEngine:
    def __init__(
        self,
        growth_features: Sequence[str],
        inflation_features: Sequence[str],
        volatility_features: Sequence[str],
        liquidity_features: Sequence[str],
        n_components: int = 4,
    ) -> None:
        self.growth_features = list(growth_features)
        self.inflation_features = list(inflation_features)
        self.volatility_features = list(volatility_features)
        self.liquidity_features = list(liquidity_features)
        self.n_components = n_components
        feature_groups = {
            "growth_features": self.growth_features,
            "inflation_features": self.inflation_features,
            "volatility_features": self.volatility_features,
            "liquidity_features": self.liquidity_features,
        }
        for group_name, features in feature_groups.items():
            if not features:
                raise ValueError(f"{group_name} must contain at least one feature")

    def fit(self, observations: Iterable[MacroObservation]) -> MacroRegimeReport:
        ordered = self._validate_and_order(observations)
        growth_scores = self._reduce_feature_group(ordered, self.growth_features)
        inflation_scores = self._reduce_feature_group(ordered, self.inflation_features)
        volatility_scores = self._reduce_feature_group(ordered, self.volatility_features)
        liquidity_scores = self._reduce_feature_group(ordered, self.liquidity_features)

        growth_inflation_points = [
            [growth_score, inflation_score]
            for growth_score, inflation_score in zip(growth_scores, inflation_scores)
        ]
        volatility_liquidity_points = [
            [volatility_score, liquidity_score]
            for volatility_score, liquidity_score in zip(volatility_scores, liquidity_scores)
        ]

        growth_inflation_axis = self._fit_axis(
            growth_inflation_points,
            ordered,
            positive_labels=("Growth↑", "Inflation↑"),
            negative_labels=("Growth↓", "Inflation↓"),
            axis_name="growth_inflation",
        )
        volatility_liquidity_axis = self._fit_axis(
            volatility_liquidity_points,
            ordered,
            positive_labels=("Volatility↑", "Liquidity↑"),
            negative_labels=("Volatility↓", "Liquidity↓"),
            axis_name="volatility_liquidity",
        )

        regime_observations: List[RegimeObservation] = []
        for index, observation in enumerate(ordered):
            gi = growth_inflation_axis["classifications"][index]
            vl = volatility_liquidity_axis["classifications"][index]
            regime_observations.append(
                RegimeObservation(
                    timestamp=_parse_date(observation.timestamp),
                    growth_inflation=gi,
                    volatility_liquidity=vl,
                    combined_label=f"{gi.label} | {vl.label}",
                )
            )

        if not regime_observations:
            raise ValueError("regime fitting produced no classifications")

        return MacroRegimeReport(
            observations=regime_observations,
            latest_forecasts={
                "growth_inflation": regime_observations[-1].growth_inflation.next_state_probabilities,
                "volatility_liquidity": regime_observations[-1].volatility_liquidity.next_state_probabilities,
            },
            transition_matrices={
                "growth_inflation": growth_inflation_axis["transition_matrix"],
                "volatility_liquidity": volatility_liquidity_axis["transition_matrix"],
            },
            nber_validation=self._validate_against_nber(ordered, growth_scores),
            asset_class_behaviour=self._map_asset_class_behaviour(ordered, regime_observations),
        )

    def _validate_and_order(self, observations: Iterable[MacroObservation]) -> List[MacroObservation]:
        ordered = sorted(observations, key=lambda observation: _parse_date(observation.timestamp))
        if not ordered:
            raise ValueError("at least one observation is required")

        required_features = (
            self.growth_features
            + self.inflation_features
            + self.volatility_features
            + self.liquidity_features
        )
        for observation in ordered:
            timestamp = _parse_date(observation.timestamp)
            available_at = _parse_date(observation.available_at)
            if available_at > timestamp:
                raise ValueError(
                    f"observation for {timestamp.isoformat()} is not point-in-time safe: available_at={available_at.isoformat()}"
                )
            for feature in required_features:
                if feature not in observation.macro_features:
                    raise ValueError(f"missing required macro feature: {feature}")
        return ordered

    def _reduce_feature_group(
        self,
        observations: Sequence[MacroObservation],
        features: Sequence[str],
    ) -> List[float]:
        matrix = [
            [float(observation.macro_features[feature]) for feature in features]
            for observation in observations
        ]
        standardized, _, _ = _standardize(matrix)
        return _principal_component_scores(standardized)

    def _fit_axis(
        self,
        points: Sequence[Sequence[float]],
        observations: Sequence[MacroObservation],
        positive_labels: tuple[str, str],
        negative_labels: tuple[str, str],
        axis_name: str,
    ) -> Dict[str, object]:
        gmm = GaussianMixtureModel(n_components=self.n_components).fit(points)
        emissions = gmm.predict_proba(points)
        decoded_states = [_argmax(probabilities) for probabilities in emissions]

        hmm = HiddenMarkovModel().fit(emissions, decoded_states, prior=gmm.weights)
        filtered, forecasts = hmm.filter_and_forecast(emissions)

        component_labels = self._build_component_labels(
            gmm.means,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )
        canonical_labels = self._ordered_unique(component_labels)
        classifications = []
        for probabilities, forecast in zip(filtered, forecasts):
            state_probabilities = self._aggregate_probabilities(probabilities, component_labels, canonical_labels)
            next_state_probabilities = self._aggregate_probabilities(forecast, component_labels, canonical_labels)
            label = max(state_probabilities, key=state_probabilities.get)
            classifications.append(
                AxisClassification(
                    label=label,
                    state_probabilities=state_probabilities,
                    confidence=max(state_probabilities.values()),
                    uncertainty=_entropy(list(state_probabilities.values())),
                    next_state_probabilities=next_state_probabilities,
                )
            )

        transition_matrix = self._aggregate_transition_matrix(
            hmm.transition_matrix,
            component_labels,
            canonical_labels,
            gmm.weights,
        )
        return {
            "name": axis_name,
            "classifications": classifications,
            "transition_matrix": transition_matrix,
        }

    def _build_component_labels(
        self,
        means: Sequence[Sequence[float]],
        positive_labels: tuple[str, str],
        negative_labels: tuple[str, str],
    ) -> List[str]:
        labels: List[str] = []
        for first_axis, second_axis in means:
            first = positive_labels[0] if first_axis >= 0.0 else negative_labels[0]
            second = positive_labels[1] if second_axis >= 0.0 else negative_labels[1]
            labels.append(f"{first} × {second}")
        return labels

    def _ordered_unique(self, labels: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        for label in labels:
            if label not in ordered:
                ordered.append(label)
        return ordered

    def _aggregate_probabilities(
        self,
        probabilities: Sequence[float],
        component_labels: Sequence[str],
        canonical_labels: Sequence[str],
    ) -> Dict[str, float]:
        aggregated = {label: 0.0 for label in canonical_labels}
        for label, probability in zip(component_labels, probabilities):
            aggregated[label] += probability
        return aggregated

    def _aggregate_transition_matrix(
        self,
        transition_matrix: Sequence[Sequence[float]],
        component_labels: Sequence[str],
        canonical_labels: Sequence[str],
        component_weights: Sequence[float],
    ) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, Dict[str, float]] = {
            source: {target: 0.0 for target in canonical_labels}
            for source in canonical_labels
        }
        source_totals = {source: 0.0 for source in canonical_labels}
        for source_index, row in enumerate(transition_matrix):
            source_label = component_labels[source_index]
            source_weight = component_weights[source_index]
            source_totals[source_label] += source_weight
            for target_index, probability in enumerate(row):
                grouped[source_label][component_labels[target_index]] += source_weight * probability

        aggregated: Dict[str, Dict[str, float]] = {}
        for source_label, targets in grouped.items():
            source_total = source_totals[source_label]
            if source_total <= 0.0:
                aggregated[source_label] = {target: 0.0 for target in canonical_labels}
            else:
                aggregated[source_label] = {
                    target: value / source_total for target, value in targets.items()
                }
        return aggregated

    def _validate_against_nber(
        self,
        observations: Sequence[MacroObservation],
        growth_scores: Sequence[float],
    ) -> Dict[str, float]:
        scored = [
            (
                growth_score < 0.0,
                bool(observation.nber_recession),
            )
            for observation, growth_score in zip(observations, growth_scores)
            if observation.nber_recession is not None
        ]
        if not scored:
            return {"evaluated_observations": 0.0}

        true_positive = sum(int(predicted and actual) for predicted, actual in scored)
        false_positive = sum(int(predicted and not actual) for predicted, actual in scored)
        false_negative = sum(int((not predicted) and actual) for predicted, actual in scored)
        accuracy = sum(int(predicted == actual) for predicted, actual in scored) / len(scored)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return {
            "evaluated_observations": float(len(scored)),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

    def _map_asset_class_behaviour(
        self,
        observations: Sequence[MacroObservation],
        classifications: Sequence[RegimeObservation],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for observation, classification in zip(observations, classifications):
            if not observation.asset_returns:
                continue
            bucket = grouped.setdefault(classification.combined_label, {})
            for asset_class, asset_return in observation.asset_returns.items():
                bucket.setdefault(asset_class, []).append(float(asset_return))

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for regime, asset_groups in grouped.items():
            summary[regime] = {}
            for asset_class, returns in asset_groups.items():
                summary[regime][asset_class] = {
                    "average_return": _mean(returns),
                    "count": float(len(returns)),
                }
        return summary
