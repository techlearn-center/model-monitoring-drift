"""
Data Drift Detection Module

Detects data drift between reference and current datasets using multiple
statistical methods: Evidently AI DataDriftPreset, Population Stability
Index (PSI), Kolmogorov-Smirnov test, Jensen-Shannon divergence, and
Wasserstein distance.

Supports both column-level and dataset-level drift analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
)
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
drift_detected_total = Counter(
    "data_drift_detected_total",
    "Total number of drift detections",
    ["feature", "method"],
)
drift_score_gauge = Gauge(
    "data_drift_score",
    "Current drift score per feature",
    ["feature", "method"],
)
drift_check_duration = Histogram(
    "data_drift_check_duration_seconds",
    "Time spent running drift checks",
)


@dataclass
class DriftResult:
    """Result of a single drift check."""

    feature: str
    method: str
    statistic: float
    p_value: Optional[float]
    is_drifted: bool
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "method": self.method,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value else None,
            "is_drifted": self.is_drifted,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DatasetDriftResult:
    """Aggregated drift result for the entire dataset."""

    total_features: int
    drifted_features: int
    drift_share: float
    is_dataset_drifted: bool
    column_results: list[DriftResult]
    dataset_drift_threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_share": round(self.drift_share, 4),
            "is_dataset_drifted": self.is_dataset_drifted,
            "dataset_drift_threshold": self.dataset_drift_threshold,
            "column_results": [r.to_dict() for r in self.column_results],
            "timestamp": self.timestamp.isoformat(),
        }


class DataDriftDetector:
    """
    Detect data drift using multiple statistical methods.

    Supports:
    - Evidently AI DataDriftPreset (column-level and dataset-level)
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov (KS) test
    - Jensen-Shannon divergence
    - Wasserstein distance

    Usage:
        detector = DataDriftDetector(
            reference_data=train_df,
            ks_threshold=0.05,
            psi_threshold=0.2,
        )
        result = detector.run_full_analysis(current_data=production_df)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        js_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1,
        dataset_drift_share_threshold: float = 0.5,
        n_bins: int = 10,
    ):
        self.reference_data = reference_data
        self.column_mapping = column_mapping or ColumnMapping()
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.dataset_drift_share_threshold = dataset_drift_share_threshold
        self.n_bins = n_bins

        self._numerical_features = reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._categorical_features = reference_data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ── Evidently AI ──────────────────────────────────────────────────────

    @drift_check_duration.time()
    def run_evidently_drift_report(
        self, current_data: pd.DataFrame, save_html: Optional[str] = None
    ) -> dict:
        """
        Run Evidently DataDriftPreset and return structured results.

        Args:
            current_data: Current production data to compare against reference.
            save_html: Optional file path to save the HTML report.

        Returns:
            Dictionary with dataset-level and column-level drift results.
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        if save_html:
            report.save_html(save_html)
            logger.info("Evidently drift report saved to %s", save_html)

        result = report.as_dict()
        logger.info(
            "Evidently drift report complete: %d metrics evaluated",
            len(result.get("metrics", [])),
        )
        return result

    def run_evidently_test_suite(
        self,
        current_data: pd.DataFrame,
        drift_share_threshold: Optional[float] = None,
    ) -> dict:
        """
        Run Evidently test suite for pass/fail drift checks.

        Args:
            current_data: Current production data.
            drift_share_threshold: Max fraction of drifted columns to pass.

        Returns:
            Test suite results as a dictionary.
        """
        threshold = drift_share_threshold or self.dataset_drift_share_threshold

        tests = [TestShareOfDriftedColumns(lt=threshold)]

        for feature in self._numerical_features + self._categorical_features:
            tests.append(TestColumnDrift(column_name=feature))

        suite = TestSuite(tests=tests)
        suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        return suite.as_dict()

    # ── Population Stability Index (PSI) ──────────────────────────────────

    def calculate_psi(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Calculate Population Stability Index between two distributions.

        PSI interpretation:
            < 0.1  : No significant shift
            0.1-0.2: Moderate shift, investigate
            > 0.2  : Significant shift, action required

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            PSI value (float).
        """
        eps = 1e-4

        # Create bins from reference data
        breakpoints = np.quantile(
            reference, np.linspace(0, 1, self.n_bins + 1)
        )
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Normalize to proportions
        ref_proportions = (ref_counts / ref_counts.sum()) + eps
        cur_proportions = (cur_counts / cur_counts.sum()) + eps

        psi = np.sum(
            (cur_proportions - ref_proportions)
            * np.log(cur_proportions / ref_proportions)
        )
        return float(psi)

    # ── Kolmogorov-Smirnov Test ───────────────────────────────────────────

    def ks_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test for distribution comparison.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            Tuple of (KS statistic, p-value).
        """
        stat, p_value = stats.ks_2samp(reference, current)
        return float(stat), float(p_value)

    # ── Jensen-Shannon Divergence ─────────────────────────────────────────

    def jensen_shannon_divergence(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.

        JSD is a symmetric, bounded (0-1) version of KL divergence.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            JSD value (float between 0 and 1).
        """
        # Bin both distributions into the same histogram
        combined = np.concatenate([reference, current])
        bins = np.histogram_bin_edges(combined, bins=self.n_bins)

        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ref_hist = ref_hist + eps
        cur_hist = cur_hist + eps

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()

        return float(jensenshannon(ref_hist, cur_hist) ** 2)

    # ── Wasserstein Distance ──────────────────────────────────────────────

    def wasserstein_distance(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """
        Calculate Wasserstein (Earth Mover's) distance between distributions.

        Measures the minimum "work" needed to transform one distribution
        into another. Also known as the Earth Mover's Distance (EMD).

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            Wasserstein distance (float).
        """
        return float(stats.wasserstein_distance(reference, current))

    # ── Column-Level Analysis ─────────────────────────────────────────────

    def check_column_drift(
        self, feature: str, current_data: pd.DataFrame
    ) -> list[DriftResult]:
        """
        Run all drift methods on a single feature column.

        Args:
            feature: Column name to check.
            current_data: Current production DataFrame.

        Returns:
            List of DriftResult objects, one per method.
        """
        ref_values = self.reference_data[feature].dropna().values
        cur_values = current_data[feature].dropna().values
        results = []

        if feature in self._numerical_features:
            # KS Test
            ks_stat, ks_p = self.ks_test(ref_values, cur_values)
            ks_drifted = ks_p < self.ks_threshold
            results.append(
                DriftResult(
                    feature=feature,
                    method="ks_test",
                    statistic=ks_stat,
                    p_value=ks_p,
                    is_drifted=ks_drifted,
                    threshold=self.ks_threshold,
                )
            )

            # PSI
            psi_val = self.calculate_psi(ref_values, cur_values)
            results.append(
                DriftResult(
                    feature=feature,
                    method="psi",
                    statistic=psi_val,
                    p_value=None,
                    is_drifted=psi_val > self.psi_threshold,
                    threshold=self.psi_threshold,
                )
            )

            # Jensen-Shannon
            js_val = self.jensen_shannon_divergence(ref_values, cur_values)
            results.append(
                DriftResult(
                    feature=feature,
                    method="jensen_shannon",
                    statistic=js_val,
                    p_value=None,
                    is_drifted=js_val > self.js_threshold,
                    threshold=self.js_threshold,
                )
            )

            # Wasserstein
            ws_val = self.wasserstein_distance(ref_values, cur_values)
            results.append(
                DriftResult(
                    feature=feature,
                    method="wasserstein",
                    statistic=ws_val,
                    p_value=None,
                    is_drifted=ws_val > self.wasserstein_threshold,
                    threshold=self.wasserstein_threshold,
                )
            )

        # Update Prometheus metrics
        for r in results:
            drift_score_gauge.labels(
                feature=r.feature, method=r.method
            ).set(r.statistic)
            if r.is_drifted:
                drift_detected_total.labels(
                    feature=r.feature, method=r.method
                ).inc()

        return results

    # ── Dataset-Level Analysis ────────────────────────────────────────────

    @drift_check_duration.time()
    def run_full_analysis(
        self, current_data: pd.DataFrame
    ) -> DatasetDriftResult:
        """
        Run all drift detection methods across all features.

        Produces a dataset-level drift verdict based on the share of
        drifted columns exceeding the dataset_drift_share_threshold.

        Args:
            current_data: Current production DataFrame.

        Returns:
            DatasetDriftResult with per-column and aggregate results.
        """
        all_results = []
        features = self._numerical_features + self._categorical_features

        for feature in features:
            if feature in current_data.columns:
                column_results = self.check_column_drift(feature, current_data)
                all_results.extend(column_results)

        # Determine dataset-level drift using KS test results
        ks_results = [r for r in all_results if r.method == "ks_test"]
        drifted_count = sum(1 for r in ks_results if r.is_drifted)
        total = len(ks_results) if ks_results else len(features)
        drift_share = drifted_count / total if total > 0 else 0.0

        dataset_result = DatasetDriftResult(
            total_features=total,
            drifted_features=drifted_count,
            drift_share=drift_share,
            is_dataset_drifted=drift_share >= self.dataset_drift_share_threshold,
            column_results=all_results,
            dataset_drift_threshold=self.dataset_drift_share_threshold,
        )

        logger.info(
            "Dataset drift analysis complete: %d/%d features drifted (%.1f%%), "
            "dataset_drifted=%s",
            drifted_count,
            total,
            drift_share * 100,
            dataset_result.is_dataset_drifted,
        )

        return dataset_result
