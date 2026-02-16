"""Stereo evaluation metrics computed from internal perception outputs."""

from dataclasses import dataclass, replace
import threading
from typing import Optional

import cv2
import numpy as np


@dataclass
class StereoEvalMetrics:
    """Latest stereo evaluation values."""

    epipolar_mean_px: Optional[float] = None
    epipolar_median_px: Optional[float] = None
    epipolar_matches: int = 0
    disparity_valid_ratio: Optional[float] = None
    depth_valid_ratio: Optional[float] = None
    depth_mean_m: Optional[float] = None


class StereoEvaluator:
    """Computes epipolar and validity metrics for each processed stereo pair."""

    def __init__(
        self,
        min_depth_m: float,
        max_depth_m: float,
        disparity_valid_threshold: float,
        orb_features: int,
        max_matches: int,
        match_ratio_test: float,
    ):
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._disparity_valid_threshold = float(disparity_valid_threshold)
        self._max_matches = int(max_matches)
        self._match_ratio_test = float(match_ratio_test)

        self._orb = cv2.ORB_create(nfeatures=int(orb_features))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._latest = StereoEvalMetrics()
        self._lock = threading.Lock()

    def update(
        self,
        left_rect_gray: np.ndarray,
        right_rect_gray: np.ndarray,
        disparity: np.ndarray,
        depth: np.ndarray,
    ) -> StereoEvalMetrics:
        """Compute metrics for one processed pair and store them as latest."""
        metrics = StereoEvalMetrics()

        if disparity.size > 0:
            disp_valid = disparity > self._disparity_valid_threshold
            metrics.disparity_valid_ratio = float(np.mean(disp_valid))

        if depth.size > 0:
            finite = np.isfinite(depth)
            valid_depth = finite & (depth >= self._min_depth_m) & (depth <= self._max_depth_m)
            metrics.depth_valid_ratio = float(np.mean(valid_depth))
            if np.any(valid_depth):
                metrics.depth_mean_m = float(np.mean(depth[valid_depth]))

        epi_mean, epi_median, epi_matches = self._compute_epipolar(left_rect_gray, right_rect_gray)
        metrics.epipolar_mean_px = epi_mean
        metrics.epipolar_median_px = epi_median
        metrics.epipolar_matches = epi_matches

        with self._lock:
            self._latest = metrics
            return replace(self._latest)

    def snapshot(self) -> StereoEvalMetrics:
        """Get a thread-safe copy of the latest metrics."""
        with self._lock:
            return replace(self._latest)

    def _compute_epipolar(
        self,
        left_rect_gray: np.ndarray,
        right_rect_gray: np.ndarray,
    ) -> tuple[Optional[float], Optional[float], int]:
        if left_rect_gray.shape[:2] != right_rect_gray.shape[:2]:
            right_rect_gray = cv2.resize(
                right_rect_gray,
                (left_rect_gray.shape[1], left_rect_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        kp1, des1 = self._orb.detectAndCompute(left_rect_gray, None)
        kp2, des2 = self._orb.detectAndCompute(right_rect_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, 0

        knn = self._matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self._match_ratio_test * second.distance:
                good_matches.append(first)

        if not good_matches:
            return None, None, 0

        good_matches = sorted(good_matches, key=lambda match: match.distance)[: self._max_matches]
        row_error = []
        for match in good_matches:
            y_left = kp1[match.queryIdx].pt[1]
            y_right = kp2[match.trainIdx].pt[1]
            row_error.append(abs(y_left - y_right))

        if not row_error:
            return None, None, 0

        dy_arr = np.asarray(row_error, dtype=np.float32)
        return float(np.mean(dy_arr)), float(np.median(dy_arr)), int(len(dy_arr))
