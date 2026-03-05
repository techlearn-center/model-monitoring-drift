"""
Retraining Trigger Module

Automatically triggers model retraining pipelines when data drift
or performance degradation is detected. Supports configurable
trigger conditions, data selection strategies, validation gates,
and integration with ML pipeline orchestrators.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import requests
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
retraining_triggered_total = Counter(
    "retraining_triggered_total",
    "Total retraining pipelines triggered",
    ["reason"],
)
retraining_success_total = Counter(
    "retraining_success_total",
    "Total successful retraining runs",
)
retraining_failed_total = Counter(
    "retraining_failed_total",
    "Total failed retraining runs",
)
retraining_duration = Histogram(
    "retraining_duration_seconds",
    "Duration of retraining runs",
)
model_version_gauge = Gauge(
    "model_current_version",
    "Current deployed model version",
)


class TriggerReason(str, Enum):
    """Reasons a retraining can be triggered."""

    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_VOLUME = "data_volume"


class RetrainingStatus(str, Enum):
    """Status of a retraining run."""

    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TriggerCondition:
    """Defines when retraining should be triggered."""

    name: str
    reason: TriggerReason
    metric: str
    condition: str  # "gt" or "lt"
    threshold: float
    min_samples: int = 1000
    cooldown_hours: float = 24.0

    _last_triggered: Optional[datetime] = field(default=None, repr=False)

    def is_met(self, value: float, sample_count: int) -> bool:
        """Check if the trigger condition is met."""
        if sample_count < self.min_samples:
            logger.debug(
                "Trigger '%s': insufficient samples (%d < %d)",
                self.name,
                sample_count,
                self.min_samples,
            )
            return False

        if self._last_triggered is not None:
            elapsed_hours = (
                datetime.utcnow() - self._last_triggered
            ).total_seconds() / 3600
            if elapsed_hours < self.cooldown_hours:
                logger.debug(
                    "Trigger '%s' in cooldown (%.1f hrs remaining)",
                    self.name,
                    self.cooldown_hours - elapsed_hours,
                )
                return False

        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        return False


@dataclass
class RetrainingRun:
    """Record of a retraining run."""

    run_id: str
    reason: TriggerReason
    trigger_name: str
    status: RetrainingStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    old_model_version: Optional[str] = None
    new_model_version: Optional[str] = None
    validation_metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "reason": self.reason.value,
            "trigger_name": self.trigger_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "old_model_version": self.old_model_version,
            "new_model_version": self.new_model_version,
            "validation_metrics": self.validation_metrics,
            "metadata": self.metadata,
        }


class RetrainingTrigger:
    """
    Manage automated model retraining based on drift and performance signals.

    Evaluates configured trigger conditions against incoming metrics.
    When a condition is met, initiates retraining through a pluggable
    pipeline runner, validates the new model, and manages the rollout.

    Usage:
        trigger = RetrainingTrigger(
            pipeline_url="http://ml-pipeline:8080/retrain",
            validation_fn=my_validation_function,
        )
        trigger.add_condition(TriggerCondition(
            name="high_drift",
            reason=TriggerReason.DATA_DRIFT,
            metric="dataset_drift_share",
            condition="gt",
            threshold=0.5,
        ))
        trigger.evaluate("dataset_drift_share", 0.65, sample_count=5000)
    """

    def __init__(
        self,
        pipeline_url: Optional[str] = None,
        validation_fn: Optional[Callable] = None,
        validation_threshold: float = 0.95,
        auto_rollback: bool = True,
        current_model_version: str = "v1.0.0",
    ):
        """
        Args:
            pipeline_url: URL of the retraining pipeline API.
            validation_fn: Function to validate the retrained model.
                           Signature: fn(model, X_val, y_val) -> dict of metrics.
            validation_threshold: Minimum performance for the new model to pass.
            auto_rollback: Whether to auto-rollback on validation failure.
            current_model_version: Version string of the currently deployed model.
        """
        self.pipeline_url = pipeline_url
        self.validation_fn = validation_fn
        self.validation_threshold = validation_threshold
        self.auto_rollback = auto_rollback
        self.current_model_version = current_model_version

        self.conditions: list[TriggerCondition] = []
        self.run_history: list[RetrainingRun] = []
        self._run_counter = 0

    def add_condition(self, condition: TriggerCondition) -> None:
        """Register a retraining trigger condition."""
        self.conditions.append(condition)
        logger.info(
            "Retraining condition added: %s (%s %s %.4f)",
            condition.name,
            condition.metric,
            condition.condition,
            condition.threshold,
        )

    def remove_condition(self, name: str) -> None:
        """Remove a trigger condition by name."""
        self.conditions = [c for c in self.conditions if c.name != name]

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        metric_name: str,
        value: float,
        sample_count: int = 0,
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[tuple] = None,
    ) -> Optional[RetrainingRun]:
        """
        Evaluate a metric against trigger conditions and start retraining
        if any condition is met.

        Args:
            metric_name: Name of the metric being evaluated.
            value: Current value of the metric.
            sample_count: Number of samples in current data.
            training_data: New training data for retraining.
            validation_data: Tuple of (X_val, y_val) for model validation.

        Returns:
            RetrainingRun if retraining was triggered, None otherwise.
        """
        for condition in self.conditions:
            if condition.metric != metric_name:
                continue

            if condition.is_met(value, sample_count):
                logger.info(
                    "Retraining trigger '%s' fired: %s=%s (threshold=%s)",
                    condition.name,
                    metric_name,
                    value,
                    condition.threshold,
                )

                run = self._start_retraining(
                    condition=condition,
                    training_data=training_data,
                    validation_data=validation_data,
                )

                condition._last_triggered = datetime.utcnow()
                retraining_triggered_total.labels(
                    reason=condition.reason.value
                ).inc()

                return run

        return None

    def evaluate_drift_result(
        self,
        drift_result: dict,
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[tuple] = None,
    ) -> Optional[RetrainingRun]:
        """
        Evaluate a DatasetDriftResult and trigger retraining if needed.

        Args:
            drift_result: Output of DataDriftDetector.run_full_analysis().to_dict()
            training_data: New training data.
            validation_data: Validation data tuple.

        Returns:
            RetrainingRun if triggered.
        """
        sample_count = drift_result.get("total_features", 0) * 100

        # Check dataset-level drift share
        run = self.evaluate(
            metric_name="dataset_drift_share",
            value=drift_result.get("drift_share", 0),
            sample_count=sample_count,
            training_data=training_data,
            validation_data=validation_data,
        )

        return run

    def evaluate_performance_alerts(
        self,
        degradation_alerts: list[dict],
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[tuple] = None,
    ) -> Optional[RetrainingRun]:
        """
        Evaluate performance degradation alerts for retraining triggers.

        Args:
            degradation_alerts: List of DegradationAlert.to_dict() results.
            training_data: New training data.
            validation_data: Validation data tuple.

        Returns:
            RetrainingRun if triggered.
        """
        for alert in degradation_alerts:
            if not alert.get("is_degraded"):
                continue

            metric_key = f"perf_{alert['metric']}"
            run = self.evaluate(
                metric_name=metric_key,
                value=alert["current_value"],
                sample_count=alert.get("sample_size", 1000),
                training_data=training_data,
                validation_data=validation_data,
            )

            if run is not None:
                return run

        return None

    # ── Retraining Pipeline ───────────────────────────────────────────────

    def _start_retraining(
        self,
        condition: TriggerCondition,
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[tuple] = None,
    ) -> RetrainingRun:
        """Start a retraining run."""
        self._run_counter += 1
        run_id = f"retrain-{self._run_counter:04d}-{int(time.time())}"

        run = RetrainingRun(
            run_id=run_id,
            reason=condition.reason,
            trigger_name=condition.name,
            status=RetrainingStatus.RUNNING,
            started_at=datetime.utcnow(),
            old_model_version=self.current_model_version,
        )

        logger.info(
            "Retraining started: run_id=%s, reason=%s, trigger=%s",
            run_id,
            condition.reason.value,
            condition.name,
        )

        try:
            if self.pipeline_url:
                run = self._run_remote_pipeline(run)
            else:
                run = self._run_local_pipeline(
                    run, training_data, validation_data
                )

            if run.status == RetrainingStatus.SUCCEEDED:
                retraining_success_total.inc()
                self.current_model_version = (
                    run.new_model_version or self.current_model_version
                )
                model_version_gauge.set(self._run_counter)
            else:
                retraining_failed_total.inc()

        except Exception as e:
            logger.error("Retraining failed: %s", e)
            run.status = RetrainingStatus.FAILED
            run.metadata["error"] = str(e)
            retraining_failed_total.inc()

        run.completed_at = datetime.utcnow()
        self.run_history.append(run)
        return run

    def _run_remote_pipeline(self, run: RetrainingRun) -> RetrainingRun:
        """Trigger retraining via remote pipeline API."""
        payload = {
            "run_id": run.run_id,
            "reason": run.reason.value,
            "current_model_version": self.current_model_version,
        }

        response = requests.post(
            self.pipeline_url, json=payload, timeout=300
        )
        response.raise_for_status()

        result = response.json()
        run.new_model_version = result.get("new_model_version")
        run.validation_metrics = result.get("validation_metrics", {})

        # Check validation
        primary_metric = run.validation_metrics.get("accuracy", 0)
        if primary_metric >= self.validation_threshold:
            run.status = RetrainingStatus.SUCCEEDED
        else:
            run.status = RetrainingStatus.FAILED
            if self.auto_rollback:
                run.status = RetrainingStatus.ROLLED_BACK
                logger.warning(
                    "New model failed validation (%.4f < %.4f), rolled back",
                    primary_metric,
                    self.validation_threshold,
                )

        return run

    def _run_local_pipeline(
        self,
        run: RetrainingRun,
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[tuple] = None,
    ) -> RetrainingRun:
        """
        Run a local retraining pipeline.

        This is a placeholder that demonstrates the pipeline structure.
        In production, replace with actual model training logic.
        """
        logger.info("Running local retraining pipeline for %s", run.run_id)

        # Step 1: Data selection and preparation
        if training_data is not None:
            logger.info(
                "Training data: %d samples, %d features",
                len(training_data),
                training_data.shape[1],
            )
        else:
            logger.warning("No training data provided for local pipeline")
            run.status = RetrainingStatus.FAILED
            run.metadata["error"] = "No training data provided"
            return run

        # Step 2: Model training (placeholder)
        run.status = RetrainingStatus.VALIDATING

        # Step 3: Validation
        if self.validation_fn and validation_data:
            X_val, y_val = validation_data
            try:
                metrics = self.validation_fn(None, X_val, y_val)
                run.validation_metrics = metrics

                primary_metric = metrics.get(
                    "accuracy", metrics.get("f1", 0)
                )
                if primary_metric >= self.validation_threshold:
                    run.status = RetrainingStatus.SUCCEEDED
                    run.new_model_version = (
                        f"v{self._run_counter + 1}.0.0"
                    )
                else:
                    run.status = RetrainingStatus.FAILED
                    if self.auto_rollback:
                        run.status = RetrainingStatus.ROLLED_BACK
            except Exception as e:
                logger.error("Validation failed: %s", e)
                run.status = RetrainingStatus.FAILED
        else:
            # No validation, assume success
            run.status = RetrainingStatus.SUCCEEDED
            run.new_model_version = f"v{self._run_counter + 1}.0.0"

        return run

    # ── History and Reporting ─────────────────────────────────────────────

    def get_run_history(self) -> list[dict]:
        """Return all retraining runs as a list of dicts."""
        return [run.to_dict() for run in self.run_history]

    def get_last_run(self) -> Optional[dict]:
        """Return the most recent retraining run."""
        if not self.run_history:
            return None
        return self.run_history[-1].to_dict()

    def get_stats(self) -> dict:
        """Return aggregate statistics about retraining runs."""
        total = len(self.run_history)
        succeeded = sum(
            1
            for r in self.run_history
            if r.status == RetrainingStatus.SUCCEEDED
        )
        failed = sum(
            1
            for r in self.run_history
            if r.status == RetrainingStatus.FAILED
        )
        rolled_back = sum(
            1
            for r in self.run_history
            if r.status == RetrainingStatus.ROLLED_BACK
        )

        return {
            "total_runs": total,
            "succeeded": succeeded,
            "failed": failed,
            "rolled_back": rolled_back,
            "success_rate": succeeded / total if total > 0 else 0.0,
            "current_model_version": self.current_model_version,
        }
