"""Range-binned RMSE analyzer for cone depth position errors."""

from dataclasses import dataclass
import math
import threading

import numpy as np


@dataclass(frozen=True)
class RangeRMSEBinStats:
    """Binned RMSE summary over fixed radial distance bins."""

    bin_centers: np.ndarray
    rmse_x: np.ndarray
    rmse_y: np.ndarray
    counts: np.ndarray


class RangeRMSEAnalyzer:
    """Accumulate cone error samples and compute fixed range-binned RMSE."""

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
        self._bin_centers = self.range_min_m + (np.arange(self.num_bins, dtype=np.float32) + 0.5) * self.bin_width_m

        self.gt_ranges = []
        self.ex_list = []
        self.ey_list = []
        self._lock = threading.Lock()

    def add_sample(self, gt_range_m: float, ex_m: float, ey_m: float) -> None:
        """Store one valid detection sample for later binning."""
        if not (
            math.isfinite(gt_range_m)
            and math.isfinite(ex_m)
            and math.isfinite(ey_m)
        ):
            return

        with self._lock:
            self.gt_ranges.append(float(gt_range_m))
            self.ex_list.append(float(ex_m))
            self.ey_list.append(float(ey_m))

    def compute_binned_rmse(self) -> RangeRMSEBinStats:
        """Compute RMSE_x/RMSE_y and counts for fixed 0-20m, 1m bins."""
        rmse_x = np.full(self.num_bins, np.nan, dtype=np.float32)
        rmse_y = np.full(self.num_bins, np.nan, dtype=np.float32)
        counts = np.zeros(self.num_bins, dtype=np.int32)

        with self._lock:
            if not self.gt_ranges:
                return RangeRMSEBinStats(
                    bin_centers=self._bin_centers,
                    rmse_x=rmse_x,
                    rmse_y=rmse_y,
                    counts=counts,
                )
            gt = np.asarray(self.gt_ranges, dtype=np.float32)
            ex = np.asarray(self.ex_list, dtype=np.float32)
            ey = np.asarray(self.ey_list, dtype=np.float32)

        valid = (
            np.isfinite(gt)
            & np.isfinite(ex)
            & np.isfinite(ey)
            & (gt >= self.range_min_m)
            & (gt <= self.range_max_m)
        )
        if not np.any(valid):
            return RangeRMSEBinStats(
                bin_centers=self._bin_centers,
                rmse_x=rmse_x,
                rmse_y=rmse_y,
                counts=counts,
            )

        gt_valid = gt[valid]
        ex_valid = ex[valid]
        ey_valid = ey[valid]

        # Clamp exact upper bound (20.0m) into the final [19, 20] bin.
        bin_indices = np.floor((gt_valid - self.range_min_m) / self.bin_width_m).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        counts = np.bincount(bin_indices, minlength=self.num_bins).astype(np.int32)
        ex_sq_sum = np.bincount(bin_indices, weights=np.square(ex_valid), minlength=self.num_bins)
        ey_sq_sum = np.bincount(bin_indices, weights=np.square(ey_valid), minlength=self.num_bins)

        non_empty = counts > 0
        rmse_x[non_empty] = np.sqrt(ex_sq_sum[non_empty] / counts[non_empty]).astype(np.float32)
        rmse_y[non_empty] = np.sqrt(ey_sq_sum[non_empty] / counts[non_empty]).astype(np.float32)

        return RangeRMSEBinStats(
            bin_centers=self._bin_centers,
            rmse_x=rmse_x,
            rmse_y=rmse_y,
            counts=counts,
        )
