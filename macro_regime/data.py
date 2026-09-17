from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class PointInTimeRecord:
    observed_at: date
    available_at: date
    values: Mapping[str, float]
    asset_returns: Mapping[str, float] = field(default_factory=dict)


class PointInTimeDataset:
    def __init__(self, records: Iterable[PointInTimeRecord]) -> None:
        self._frame = self._build_frame(records)
        if self._frame.empty:
            raise ValueError("PointInTimeDataset requires at least one record.")

    @staticmethod
    def _build_frame(records: Iterable[PointInTimeRecord]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in records:
            row: dict[str, object] = {
                "observed_at": pd.Timestamp(record.observed_at),
                "available_at": pd.Timestamp(record.available_at),
            }
            row.update(record.values)
            for asset, value in record.asset_returns.items():
                row[f"asset::{asset}"] = value
            rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["observed_at", "available_at"]).reset_index(drop=True)

    @property
    def feature_names(self) -> list[str]:
        return [column for column in self._frame.columns if column not in {"observed_at", "available_at"} and not column.startswith("asset::")]

    @property
    def asset_names(self) -> list[str]:
        return [column.removeprefix("asset::") for column in self._frame.columns if column.startswith("asset::")]

    @property
    def observed_dates(self) -> list[date]:
        return [ts.date() for ts in self._frame["observed_at"].drop_duplicates().sort_values()]

    def snapshot(self, as_of: date) -> pd.DataFrame:
        as_of_ts = pd.Timestamp(as_of)
        eligible = self._frame[self._frame["available_at"] <= as_of_ts]
        if eligible.empty:
            return eligible.copy()
        latest = (
            eligible.sort_values(["observed_at", "available_at"])
            .groupby("observed_at", as_index=False)
            .tail(1)
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
        return latest

    def time_series(self, as_of: date | None = None) -> pd.DataFrame:
        if as_of is not None:
            return (
                self.snapshot(as_of)
                .loc[lambda frame: frame["observed_at"] <= pd.Timestamp(as_of)]
                .sort_values("observed_at")
                .reset_index(drop=True)
            )

        if self._frame.empty:
            return pd.DataFrame(columns=self._frame.columns)
        return (
            self._frame.groupby("observed_at", as_index=False)
            .tail(1)
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
