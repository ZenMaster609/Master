"""Range-binned RMSE analyzer for cone position errors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Dict, Optional

import numpy as np


_SOURCE_ORDER = ("monocular", "stereo", "lidar")


@dataclass(frozen=True)
class RangeRMSEBinStats:
    """Binned RMSE summary over fixed radial distance bins."""

    bin_centers: np.ndarray
    total_counts: np.ndarray
    source_rmse: Dict[str, np.ndarray]
    correct_class_count: int
    incorrect_class_count: int


class RangeRMSEAnalyzer:
    """Accumulate cone error samples and compute fixed range-binned RMSE."""

    SOURCE_ORDER = _SOURCE_ORDER

    def __init__(
        self,
        range_min_m: float = 0.0,
        range_max_m: float = 20.0,
        bin_width_m: float = 1.0,
    ):
        self.range_min_m = float(range_min_m)
        self.range_max_m = float(range_max_m)
        self.bin_width_m = float(bin_width_m)

        span = self.range_max_m - self.range_min_m
        self.num_bins = max(1, int(round(span / self.bin_width_m)))
        self._bin_centers = self.range_min_m + (
            (np.arange(self.num_bins, dtype=np.float32) + 0.5) * self.bin_width_m
        )

        self._samples: list[tuple[str, float, float]] = []
        self.correct_class_count = 0
        self.incorrect_class_count = 0
        self._lock = threading.Lock()

    def add_sample(
        self,
        source: str,
        gt_range_m: float,
        error_m: float,
        predicted_class_id: Optional[int] = None,
        ground_truth_class_id: Optional[int] = None,
    ) -> None:
        """Store one valid detection sample for later binning."""
        if not math.isfinite(gt_range_m) or not math.isfinite(error_m):
            return

        source_name = str(source).strip().lower()
        if not source_name:
            return

        in_range = self.range_min_m <= float(gt_range_m) <= self.range_max_m
        with self._lock:
            self._samples.append((source_name, float(gt_range_m), float(error_m)))
            if (
                in_range
                and predicted_class_id is not None
                and ground_truth_class_id is not None
            ):
                if int(predicted_class_id) == int(ground_truth_class_id):
                    self.correct_class_count += 1
                else:
                    self.incorrect_class_count += 1

    def compute_binned_rmse(self) -> RangeRMSEBinStats:
        """Compute combined RMSE and counts for fixed 0-20m, 1m bins."""
        total_counts = np.zeros(self.num_bins, dtype=np.int32)

        with self._lock:
            samples = list(self._samples)
            correct_class_count = int(self.correct_class_count)
            incorrect_class_count = int(self.incorrect_class_count)

        present_sources = self._ordered_sources({sample[0] for sample in samples})
        source_rmse = {
            source: np.full(self.num_bins, np.nan, dtype=np.float32)
            for source in present_sources
        }

        if not samples:
            return RangeRMSEBinStats(
                bin_centers=self._bin_centers,
                total_counts=total_counts,
                source_rmse=source_rmse,
                correct_class_count=correct_class_count,
                incorrect_class_count=incorrect_class_count,
            )

        gt = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        error = np.asarray([sample[2] for sample in samples], dtype=np.float32)
        source_arr = np.asarray([sample[0] for sample in samples], dtype=object)

        valid = (
            np.isfinite(gt)
            & np.isfinite(error)
            & (gt >= self.range_min_m)
            & (gt <= self.range_max_m)
        )
        if not np.any(valid):
            return RangeRMSEBinStats(
                bin_centers=self._bin_centers,
                total_counts=total_counts,
                source_rmse=source_rmse,
                correct_class_count=correct_class_count,
                incorrect_class_count=incorrect_class_count,
            )

        gt_valid = gt[valid]
        error_valid = error[valid]
        source_valid = source_arr[valid]

        bin_indices = np.floor(
            (gt_valid - self.range_min_m) / self.bin_width_m
        ).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        total_counts = np.bincount(bin_indices, minlength=self.num_bins).astype(
            np.int32
        )

        for source in self._ordered_sources(set(source_valid.tolist())):
            mask = source_valid == source
            if not np.any(mask):
                continue
            source_bins = bin_indices[mask]
            source_error = error_valid[mask]
            counts = np.bincount(source_bins, minlength=self.num_bins).astype(
                np.int32
            )
            error_sq_sum = np.bincount(
                source_bins,
                weights=np.square(source_error),
                minlength=self.num_bins,
            )
            rmse = np.full(self.num_bins, np.nan, dtype=np.float32)
            non_empty = counts > 0
            rmse[non_empty] = np.sqrt(
                error_sq_sum[non_empty] / counts[non_empty]
            ).astype(np.float32)
            source_rmse[source] = rmse

        return RangeRMSEBinStats(
            bin_centers=self._bin_centers,
            total_counts=total_counts,
            source_rmse=source_rmse,
            correct_class_count=correct_class_count,
            incorrect_class_count=incorrect_class_count,
        )

    @classmethod
    def _ordered_sources(cls, sources: set[str]) -> list[str]:
        ordered = [source for source in cls.SOURCE_ORDER if source in sources]
        extras = sorted(source for source in sources if source not in cls.SOURCE_ORDER)
        return ordered + extras
