"""
Alerting Module

Sends alerts when drift or performance degradation exceeds configured
thresholds. Supports multiple channels: Prometheus Alertmanager,
Slack webhooks, PagerDuty, email, and custom webhook endpoints.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import requests
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
alerts_sent_total = Counter(
    "monitoring_alerts_sent_total",
    "Total alerts sent",
    ["channel", "severity"],
)
alerts_failed_total = Counter(
    "monitoring_alerts_failed_total",
    "Total alerts that failed to send",
    ["channel"],
)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Supported alerting channels."""

    ALERTMANAGER = "alertmanager"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Structured alert payload."""

    title: str
    description: str
    severity: AlertSeverity
    source: str
    labels: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source": self.source,
            "labels": self.labels,
            "annotations": self.annotations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlertRule:
    """Defines when and how to alert."""

    name: str
    metric: str
    condition: str  # "gt" (greater than) or "lt" (less than)
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    channels: list[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.LOG]
    )
    cooldown_minutes: int = 30

    _last_fired: Optional[datetime] = field(default=None, repr=False)

    def should_fire(self, value: float) -> bool:
        """Check if the rule condition is met."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        return False

    def is_in_cooldown(self) -> bool:
        """Check if the rule is still in cooldown period."""
        if self._last_fired is None:
            return False
        elapsed = (datetime.utcnow() - self._last_fired).total_seconds()
        return elapsed < (self.cooldown_minutes * 60)


class AlertManager:
    """
    Manage alert rules and dispatch alerts to configured channels.

    Evaluates incoming metric values against configured rules and
    sends alerts through the appropriate channels when thresholds
    are breached.

    Usage:
        manager = AlertManager(
            alertmanager_url="http://localhost:9093",
            slack_webhook_url="https://hooks.slack.com/...",
        )
        manager.add_rule(AlertRule(
            name="high_drift",
            metric="drift_score",
            condition="gt",
            threshold=0.2,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.ALERTMANAGER],
        ))
        manager.evaluate_metric("drift_score", 0.35)
    """

    def __init__(
        self,
        alertmanager_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        pagerduty_api_key: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.alertmanager_url = alertmanager_url
        self.slack_webhook_url = slack_webhook_url
        self.pagerduty_api_key = pagerduty_api_key
        self.webhook_url = webhook_url
        self.rules: list[AlertRule] = []
        self.alert_history: list[dict] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule."""
        self.rules.append(rule)
        logger.info(
            "Alert rule added: %s (%s %s %.4f)",
            rule.name,
            rule.metric,
            rule.condition,
            rule.threshold,
        )

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        self.rules = [r for r in self.rules if r.name != name]

    # ── Rule Evaluation ───────────────────────────────────────────────────

    def evaluate_metric(self, metric_name: str, value: float) -> list[Alert]:
        """
        Evaluate a metric value against all matching rules.

        Args:
            metric_name: Name of the metric being evaluated.
            value: Current value of the metric.

        Returns:
            List of Alert objects that were fired.
        """
        fired_alerts = []

        for rule in self.rules:
            if rule.metric != metric_name:
                continue
            if not rule.should_fire(value):
                continue
            if rule.is_in_cooldown():
                logger.debug(
                    "Rule '%s' is in cooldown, skipping", rule.name
                )
                continue

            alert = Alert(
                title=f"[{rule.severity.value.upper()}] {rule.name}",
                description=(
                    f"Metric '{metric_name}' has value {value:.4f} which "
                    f"{'exceeds' if rule.condition == 'gt' else 'is below'} "
                    f"threshold {rule.threshold:.4f}"
                ),
                severity=rule.severity,
                source="model-monitoring-drift",
                labels={
                    "metric": metric_name,
                    "rule": rule.name,
                    "severity": rule.severity.value,
                },
                annotations={
                    "current_value": str(value),
                    "threshold": str(rule.threshold),
                    "condition": rule.condition,
                },
            )

            self._dispatch_alert(alert, rule.channels)
            rule._last_fired = datetime.utcnow()
            fired_alerts.append(alert)
            self.alert_history.append(alert.to_dict())

        return fired_alerts

    def evaluate_drift_result(self, drift_result: dict) -> list[Alert]:
        """
        Evaluate a DatasetDriftResult and fire alerts for drifted features.

        Args:
            drift_result: Output of DataDriftDetector.run_full_analysis().to_dict()

        Returns:
            List of fired alerts.
        """
        alerts = []

        # Dataset-level drift
        if drift_result.get("is_dataset_drifted"):
            alerts.extend(
                self.evaluate_metric(
                    "dataset_drift_share", drift_result["drift_share"]
                )
            )

        # Column-level drift
        for col_result in drift_result.get("column_results", []):
            if col_result.get("is_drifted"):
                metric_key = f"drift_{col_result['method']}_{col_result['feature']}"
                alerts.extend(
                    self.evaluate_metric(metric_key, col_result["statistic"])
                )

        return alerts

    def evaluate_performance_alerts(
        self, degradation_alerts: list[dict]
    ) -> list[Alert]:
        """
        Evaluate performance degradation alerts.

        Args:
            degradation_alerts: List of DegradationAlert.to_dict() results.

        Returns:
            List of fired alerts.
        """
        alerts = []
        for deg_alert in degradation_alerts:
            if deg_alert.get("is_degraded"):
                alerts.extend(
                    self.evaluate_metric(
                        f"perf_{deg_alert['metric']}",
                        deg_alert["current_value"],
                    )
                )
        return alerts

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _dispatch_alert(
        self, alert: Alert, channels: list[AlertChannel]
    ) -> None:
        """Send alert to all specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    self._send_log_alert(alert)
                elif channel == AlertChannel.ALERTMANAGER:
                    self._send_alertmanager(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                elif channel == AlertChannel.PAGERDUTY:
                    self._send_pagerduty(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)

                alerts_sent_total.labels(
                    channel=channel.value, severity=alert.severity.value
                ).inc()
            except Exception as e:
                alerts_failed_total.labels(channel=channel.value).inc()
                logger.error(
                    "Failed to send alert via %s: %s", channel.value, e
                )

    def _send_log_alert(self, alert: Alert) -> None:
        """Send alert to application logs."""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }
        log_fn = log_method.get(alert.severity, logger.warning)
        log_fn("ALERT: %s - %s", alert.title, alert.description)

    def _send_alertmanager(self, alert: Alert) -> None:
        """Send alert to Prometheus Alertmanager."""
        if not self.alertmanager_url:
            logger.warning("Alertmanager URL not configured")
            return

        payload = [
            {
                "labels": {
                    "alertname": alert.title,
                    "severity": alert.severity.value,
                    **alert.labels,
                },
                "annotations": {
                    "summary": alert.description,
                    **alert.annotations,
                },
                "generatorURL": "http://model-monitoring-drift/alerts",
            }
        ]

        response = requests.post(
            f"{self.alertmanager_url}/api/v1/alerts",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Alert sent to Alertmanager: %s", alert.title)

    def _send_slack(self, alert: Alert) -> None:
        """Send alert to Slack via webhook."""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        severity_emoji = {
            AlertSeverity.INFO: "information_source",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "rotating_light",
        }
        emoji = severity_emoji.get(alert.severity, "bell")

        payload = {
            "text": f":{emoji}: *{alert.title}*\n{alert.description}",
            "attachments": [
                {
                    "color": {
                        AlertSeverity.INFO: "#36a64f",
                        AlertSeverity.WARNING: "#ff9900",
                        AlertSeverity.CRITICAL: "#ff0000",
                    }.get(alert.severity, "#cccccc"),
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in alert.annotations.items()
                    ],
                }
            ],
        }

        response = requests.post(
            self.slack_webhook_url, json=payload, timeout=10
        )
        response.raise_for_status()
        logger.info("Alert sent to Slack: %s", alert.title)

    def _send_pagerduty(self, alert: Alert) -> None:
        """Send alert to PagerDuty."""
        if not self.pagerduty_api_key:
            logger.warning("PagerDuty API key not configured")
            return

        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
        }

        payload = {
            "routing_key": self.pagerduty_api_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{alert.title}: {alert.description}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": alert.source,
                "custom_details": {**alert.labels, **alert.annotations},
            },
        }

        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Alert sent to PagerDuty: %s", alert.title)

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to a custom webhook endpoint."""
        if not self.webhook_url:
            logger.warning("Webhook URL not configured")
            return

        response = requests.post(
            self.webhook_url, json=alert.to_dict(), timeout=10
        )
        response.raise_for_status()
        logger.info("Alert sent to webhook: %s", alert.title)
