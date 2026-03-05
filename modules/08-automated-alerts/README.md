# Module 08: Automated Alerts and Actions

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 07 completed, Docker installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Set up Prometheus for collecting ML monitoring metrics
- Build Grafana dashboards for visualizing drift and performance
- Configure Alertmanager with routing rules and severity levels
- Integrate alerting with Slack, PagerDuty, and custom webhooks
- Design an alerting strategy with cooldowns, escalation, and runbooks

---

## Concepts

### The Alerting Stack

```
+-------------------+     +------------------+     +------------------+
| Monitoring App    |     | Prometheus       |     | Alertmanager     |
| (FastAPI + Python)|---->| (metrics store)  |---->| (routing/dedup)  |
| Exposes /metrics  |     | Evaluates rules  |     | Sends alerts     |
+-------------------+     +------------------+     +--------+---------+
                                                            |
                          +------------------+     +--------v---------+
                          | Grafana          |     | Alert Channels   |
                          | (dashboards)     |     | - Slack          |
                          | Queries Prometheus|     | - PagerDuty     |
                          +------------------+     | - Email          |
                                                   | - Webhooks       |
                                                   +------------------+
```

### Prometheus Metrics for ML Monitoring

Prometheus supports four metric types. Here is how each applies to model monitoring:

| Metric Type | ML Use Case | Example |
|---|---|---|
| **Counter** | Count events that only go up | `drift_detected_total`, `predictions_total` |
| **Gauge** | Track values that go up and down | `model_accuracy`, `drift_score` |
| **Histogram** | Track distributions of values | `prediction_latency_seconds` |
| **Summary** | Track quantiles of values | `prediction_confidence_p50`, `_p99` |

```python
from prometheus_client import Counter, Gauge, Histogram

# Define ML monitoring metrics
predictions_total = Counter(
    "model_predictions_total",
    "Total predictions served",
    ["model_version", "endpoint"],
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy",
    ["model_name"],
)

drift_score = Gauge(
    "data_drift_score",
    "Current data drift score per feature",
    ["feature", "method"],
)

prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
```

### Prometheus Alert Rules

Alert rules define conditions that trigger alerts. They are evaluated on a schedule by Prometheus.

```yaml
# config/prometheus/alert_rules.yml
groups:
  - name: drift_alerts
    rules:
      - alert: DataDriftDetected
        expr: data_drift_score > 0.2
        for: 5m        # Must be true for 5 minutes
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected on {{ $labels.feature }}"
          description: "Drift score {{ $value }} exceeds threshold 0.2"

      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.85
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below 85%"
          runbook: "https://wiki.internal/runbooks/model-accuracy-drop"
```

### Alertmanager Routing

Alertmanager handles deduplication, grouping, silencing, and routing of alerts to the right channels.

```yaml
# config/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ["alertname", "severity"]
  group_wait: 10s      # Wait before sending first notification
  group_interval: 10s  # Wait between notifications for same group
  repeat_interval: 1h  # Resend if still firing
  receiver: "default"
  routes:
    - match:
        severity: critical
      receiver: "critical-alerts"
      continue: false
    - match:
        severity: warning
      receiver: "warning-alerts"

receivers:
  - name: "default"
    webhook_configs:
      - url: "http://app:8000/api/alerts/webhook"

  - name: "critical-alerts"
    pagerduty_configs:
      - service_key: "$PAGERDUTY_KEY"
    slack_configs:
      - api_url: "$SLACK_WEBHOOK"
        channel: "#ml-alerts-critical"

  - name: "warning-alerts"
    slack_configs:
      - api_url: "$SLACK_WEBHOOK"
        channel: "#ml-alerts"
```

### Grafana Dashboards for ML Monitoring

Key panels for an ML monitoring dashboard:

| Panel | Visualization | Query |
|---|---|---|
| Model Accuracy Over Time | Time series line chart | `model_accuracy` |
| Drift Score Heatmap | Heatmap by feature | `data_drift_score` |
| Alert Timeline | State timeline | Alertmanager API |
| Prediction Volume | Bar chart | `rate(model_predictions_total[5m])` |
| Prediction Latency | Histogram | `histogram_quantile(0.95, prediction_latency)` |
| Retraining History | Annotations | `retraining_triggered_total` |

### Alerting Best Practices

#### 1. Severity Levels

| Level | When to Use | Response Time | Channel |
|---|---|---|---|
| **INFO** | FYI, no action needed | Next business day | Dashboard, email digest |
| **WARNING** | Investigate soon | Within 4 hours | Slack channel |
| **CRITICAL** | Immediate action required | Within 15 minutes | PagerDuty, phone call |

#### 2. Avoid Alert Fatigue

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Too many alerts | Team ignores all of them | Tune thresholds, use cooldowns |
| No grouping | Same issue triggers 50 alerts | Group by alertname, service |
| No escalation | Critical issues go unnoticed | Route critical to PagerDuty |
| No runbooks | Engineers do not know what to do | Link runbook in annotation |
| No auto-resolve | Stale alerts pile up | Set resolve_timeout |

#### 3. Cooldown Periods

Prevent the same alert from firing repeatedly:

```python
from src.monitoring.alerting import AlertManager, AlertRule, AlertSeverity, AlertChannel

manager = AlertManager(
    slack_webhook_url="https://hooks.slack.com/...",
    alertmanager_url="http://localhost:9093",
)

manager.add_rule(AlertRule(
    name="high_drift",
    metric="drift_score",
    condition="gt",
    threshold=0.2,
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.SLACK, AlertChannel.ALERTMANAGER],
    cooldown_minutes=30,  # Do not re-fire for 30 minutes
))
```

---

## Hands-On Lab

### Exercise 1: Deploy the Monitoring Stack

**Goal:** Start Prometheus, Grafana, and Alertmanager with Docker Compose.

```bash
# Start the monitoring stack
docker compose up -d prometheus grafana alertmanager pushgateway

# Verify services are running
docker compose ps

# Access the UIs
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000 (admin/changeme)"
echo "Alertmanager: http://localhost:9093"
```

### Exercise 2: Configure Alert Rules

**Goal:** Set up alert rules and verify they trigger.

```python
from src.monitoring.alerting import (
    AlertManager, AlertRule, AlertSeverity, AlertChannel
)

# Initialize alert manager
manager = AlertManager(
    alertmanager_url="http://localhost:9093",
)

# Add drift alert rules
manager.add_rule(AlertRule(
    name="data_drift_warning",
    metric="dataset_drift_share",
    condition="gt",
    threshold=0.3,
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.LOG, AlertChannel.ALERTMANAGER],
    cooldown_minutes=15,
))

manager.add_rule(AlertRule(
    name="data_drift_critical",
    metric="dataset_drift_share",
    condition="gt",
    threshold=0.5,
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.LOG, AlertChannel.ALERTMANAGER],
    cooldown_minutes=5,
))

manager.add_rule(AlertRule(
    name="accuracy_degradation",
    metric="perf_accuracy",
    condition="lt",
    threshold=0.85,
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.LOG, AlertChannel.ALERTMANAGER],
    cooldown_minutes=10,
))

# Simulate metrics
alerts = manager.evaluate_metric("dataset_drift_share", 0.65)
print(f"Fired {len(alerts)} alerts:")
for alert in alerts:
    print(f"  [{alert.severity.value}] {alert.title}: {alert.description}")
```

### Exercise 3: Build a Grafana Dashboard

**Goal:** Create a Grafana dashboard that visualizes drift and performance metrics.

1. Open Grafana at `http://localhost:3000`
2. Log in with `admin` / `changeme`
3. Go to Dashboards -> New Dashboard
4. Add these panels:

**Panel 1: Model Accuracy**
```
Query: model_accuracy
Visualization: Time series
Thresholds: 0.85 (warning), 0.70 (critical)
```

**Panel 2: Drift Scores by Feature**
```
Query: data_drift_score
Legend: {{feature}} ({{method}})
Visualization: Time series
```

**Panel 3: Alerts Fired**
```
Query: increase(monitoring_alerts_sent_total[1h])
Visualization: Bar chart
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No cooldown on alerts | Hundreds of duplicate alerts | Set cooldown_minutes on every rule |
| All alerts go to same channel | Critical alerts get lost in noise | Route by severity to different channels |
| No runbook links | Engineers do not know how to respond | Add runbook URL in alert annotations |
| Alerting on raw metrics | Noisy, triggers on brief spikes | Use `for: 5m` clause in Prometheus rules |
| No alert testing | Discover alerting is broken during an incident | Regularly test alert pipeline end-to-end |

---

## Self-Check Questions

1. What are the four Prometheus metric types, and when would you use each for ML monitoring?
2. How does Alertmanager handle alert deduplication and grouping?
3. What is the purpose of the `for` clause in a Prometheus alert rule?
4. How would you design an alerting strategy to avoid alert fatigue?
5. What information should a good ML monitoring alert include?

---

## You Know You Have Completed This Module When...

- [ ] Prometheus, Grafana, and Alertmanager are running via Docker Compose
- [ ] You have configured alert rules for drift and performance degradation
- [ ] You have built a Grafana dashboard with key ML monitoring panels
- [ ] Validation script passes: `bash modules/08-automated-alerts/validation/validate.sh`
- [ ] You understand alert routing, severity levels, and cooldowns

---

## Troubleshooting

### Common Issues

**Issue: Prometheus cannot scrape the app**
```bash
# Check that the app is exposing /metrics
curl http://localhost:8000/metrics

# Check Prometheus targets page
# http://localhost:9090/targets
```

**Issue: Grafana cannot connect to Prometheus**
- Verify the datasource URL uses the Docker network name (e.g., `http://prometheus:9090`)
- Check that both containers are on the same Docker network

**Issue: Alertmanager not sending alerts**
```bash
# Check Alertmanager configuration
docker compose logs alertmanager

# Verify alerts are firing in Prometheus
# http://localhost:9090/alerts
```

---

**Next: [Module 09 -- Retraining Triggers and Pipelines ->](../09-retraining-triggers/)**
