"""
Model Performance Tracking Module

Tracks model performance metrics over time and detects degradation.
Monitors accuracy, precision, recall, F1 score, and AUC-ROC with
configurable thresholds and sliding window analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
model_accuracy_gauge = Gauge("model_accuracy", "Current model accuracy")
model_f1_gauge = Gauge("model_f1_score", "Current model F1 score")
model_auc_gauge = Gauge("model_auc_roc", "Current model AUC-ROC")
model_precision_gauge = Gauge("model_precision", "Current model precision")
model_recall_gauge = Gauge("model_recall", "Current model recall")
performance_degradation_counter = Counter(
    "model_performance_degradation_total",
    "Total performance degradation events detected",
    ["metric"],
)
performance_check_duration = Histogram(
    "model_performance_check_duration_seconds",
    "Time spent computing performance metrics",
)


@dataclass
class PerformanceSnapshot:
    """A point-in-time capture of model performance metrics."""

    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    log_loss_val: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    sample_size: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "accuracy": round(self.accuracy, 6) if self.accuracy else None,
            "precision": round(self.precision, 6) if self.precision else None,
            "recall": round(self.recall, 6) if self.recall else None,
            "f1": round(self.f1, 6) if self.f1 else None,
            "auc_roc": round(self.auc_roc, 6) if self.auc_roc else None,
            "log_loss": (
                round(self.log_loss_val, 6) if self.log_loss_val else None
            ),
            "mse": round(self.mse, 6) if self.mse else None,
            "mae": round(self.mae, 6) if self.mae else None,
            "r2": round(self.r2, 6) if self.r2 else None,
            "sample_size": self.sample_size,
        }


@dataclass
class DegradationAlert:
    """Alert triggered when performance drops below threshold."""

    metric: str
    baseline_value: float
    current_value: float
    degradation: float
    threshold: float
    is_degraded: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline_value": round(self.baseline_value, 6),
            "current_value": round(self.current_value, 6),
            "degradation": round(self.degradation, 6),
            "threshold": self.threshold,
            "is_degraded": self.is_degraded,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelPerformanceTracker:
    """
    Track and monitor ML model performance over time.

    Computes classification or regression metrics, maintains a history
    of snapshots, and raises degradation alerts when metrics drop
    below configured thresholds relative to the baseline.

    Usage:
        tracker = ModelPerformanceTracker(
            task_type="classification",
            degradation_threshold=0.05,
        )
        tracker.set_baseline(y_true_train, y_pred_train, y_proba_train)
        alerts = tracker.evaluate(y_true_prod, y_pred_prod, y_proba_prod)
    """

    def __init__(
        self,
        task_type: str = "classification",
        degradation_threshold: float = 0.05,
        window_size: int = 10,
        average: str = "weighted",
    ):
        """
        Args:
            task_type: "classification" or "regression".
            degradation_threshold: Relative drop to trigger a degradation alert.
            window_size: Number of recent snapshots for rolling analysis.
            average: Averaging method for multiclass metrics.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.task_type = task_type
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
        self.average = average

        self.baseline: Optional[PerformanceSnapshot] = None
        self.history: list[PerformanceSnapshot] = []

    # ── Metric Computation ────────────────────────────────────────────────

    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceSnapshot:
        """Compute classification metrics and return a snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(
                precision_score(
                    y_true, y_pred, average=self.average, zero_division=0
                )
            ),
            recall=float(
                recall_score(
                    y_true, y_pred, average=self.average, zero_division=0
                )
            ),
            f1=float(
                f1_score(
                    y_true, y_pred, average=self.average, zero_division=0
                )
            ),
            sample_size=len(y_true),
        )

        if y_proba is not None:
            try:
                if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                    proba = (
                        y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    )
                    snapshot.auc_roc = float(roc_auc_score(y_true, proba))
                else:
                    snapshot.auc_roc = float(
                        roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class="ovr",
                            average=self.average,
                        )
                    )
                snapshot.log_loss_val = float(log_loss(y_true, y_proba))
            except ValueError as e:
                logger.warning("Could not compute AUC/log_loss: %s", e)

        return snapshot

    def _compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceSnapshot:
        """Compute regression metrics and return a snapshot."""
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            mse=float(mean_squared_error(y_true, y_pred)),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
            sample_size=len(y_true),
        )

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceSnapshot:
        """
        Compute metrics based on the configured task type.

        Args:
            y_true: Ground truth labels/values.
            y_pred: Model predictions.
            y_proba: Prediction probabilities (classification only).

        Returns:
            PerformanceSnapshot with computed metrics.
        """
        if self.task_type == "classification":
            return self._compute_classification_metrics(
                y_true, y_pred, y_proba
            )
        else:
            return self._compute_regression_metrics(y_true, y_pred)

    # ── Baseline ──────────────────────────────────────────────────────────

    def set_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceSnapshot:
        """
        Establish the baseline performance from training/validation data.

        Args:
            y_true: Ground truth labels/values.
            y_pred: Model predictions.
            y_proba: Prediction probabilities (classification only).

        Returns:
            Baseline PerformanceSnapshot.
        """
        self.baseline = self.compute_metrics(y_true, y_pred, y_proba)
        logger.info(
            "Baseline set - accuracy=%.4f, f1=%.4f, auc=%.4f",
            self.baseline.accuracy or 0,
            self.baseline.f1 or 0,
            self.baseline.auc_roc or 0,
        )
        return self.baseline

    # ── Degradation Detection ─────────────────────────────────────────────

    def _check_degradation(
        self, metric_name: str, baseline_val: Optional[float], current_val: Optional[float]
    ) -> Optional[DegradationAlert]:
        """Check if a metric has degraded beyond the threshold."""
        if baseline_val is None or current_val is None:
            return None

        # For error metrics (MSE, MAE, log_loss), an increase is degradation
        if metric_name in ("mse", "mae", "log_loss"):
            degradation = (current_val - baseline_val) / max(
                abs(baseline_val), 1e-10
            )
            is_degraded = degradation > self.degradation_threshold
        else:
            # For accuracy, F1, AUC, precision, recall, R2 -- a decrease is degradation
            degradation = (baseline_val - current_val) / max(
                abs(baseline_val), 1e-10
            )
            is_degraded = degradation > self.degradation_threshold

        if is_degraded:
            performance_degradation_counter.labels(metric=metric_name).inc()
            logger.warning(
                "Performance degradation detected: %s dropped from %.4f to %.4f "
                "(%.2f%% degradation, threshold=%.2f%%)",
                metric_name,
                baseline_val,
                current_val,
                degradation * 100,
                self.degradation_threshold * 100,
            )

        return DegradationAlert(
            metric=metric_name,
            baseline_value=baseline_val,
            current_value=current_val,
            degradation=degradation,
            threshold=self.degradation_threshold,
            is_degraded=is_degraded,
        )

    # ── Evaluate ──────────────────────────────────────────────────────────

    @performance_check_duration.time()
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> list[DegradationAlert]:
        """
        Evaluate current model performance against the baseline.

        Computes current metrics, stores in history, and checks each
        metric for degradation relative to the baseline.

        Args:
            y_true: Ground truth labels/values.
            y_pred: Model predictions.
            y_proba: Prediction probabilities (classification only).

        Returns:
            List of DegradationAlert objects (one per metric).
        """
        if self.baseline is None:
            raise RuntimeError(
                "Baseline not set. Call set_baseline() first."
            )

        current = self.compute_metrics(y_true, y_pred, y_proba)
        self.history.append(current)

        # Update Prometheus gauges
        if current.accuracy is not None:
            model_accuracy_gauge.set(current.accuracy)
        if current.f1 is not None:
            model_f1_gauge.set(current.f1)
        if current.auc_roc is not None:
            model_auc_gauge.set(current.auc_roc)
        if current.precision is not None:
            model_precision_gauge.set(current.precision)
        if current.recall is not None:
            model_recall_gauge.set(current.recall)

        # Check degradation for every metric
        alerts = []
        metric_pairs = [
            ("accuracy", self.baseline.accuracy, current.accuracy),
            ("precision", self.baseline.precision, current.precision),
            ("recall", self.baseline.recall, current.recall),
            ("f1", self.baseline.f1, current.f1),
            ("auc_roc", self.baseline.auc_roc, current.auc_roc),
            ("log_loss", self.baseline.log_loss_val, current.log_loss_val),
            ("mse", self.baseline.mse, current.mse),
            ("mae", self.baseline.mae, current.mae),
            ("r2", self.baseline.r2, current.r2),
        ]

        for metric_name, baseline_val, current_val in metric_pairs:
            alert = self._check_degradation(
                metric_name, baseline_val, current_val
            )
            if alert is not None:
                alerts.append(alert)

        return alerts

    # ── Rolling Analysis ──────────────────────────────────────────────────

    def get_rolling_metrics(self) -> pd.DataFrame:
        """
        Return a DataFrame of the most recent performance snapshots.

        Uses the configured window_size to limit the history window.

        Returns:
            DataFrame with timestamp-indexed performance metrics.
        """
        recent = self.history[-self.window_size :]
        if not recent:
            return pd.DataFrame()

        records = [s.to_dict() for s in recent]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")

    def get_trend(self, metric: str) -> dict:
        """
        Calculate the trend direction and magnitude for a given metric.

        Args:
            metric: Name of the metric (e.g., "accuracy", "f1").

        Returns:
            Dictionary with slope, direction, and recent values.
        """
        df = self.get_rolling_metrics()
        if df.empty or metric not in df.columns:
            return {"slope": 0.0, "direction": "stable", "values": []}

        values = df[metric].dropna().values
        if len(values) < 2:
            return {"slope": 0.0, "direction": "stable", "values": values.tolist()}

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "improving" if metric not in ("mse", "mae", "log_loss") else "degrading"
        else:
            direction = "degrading" if metric not in ("mse", "mae", "log_loss") else "improving"

        return {
            "slope": float(slope),
            "direction": direction,
            "values": values.tolist(),
        }
