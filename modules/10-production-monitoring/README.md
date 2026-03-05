# Module 10: Production Monitoring System -- End-to-End Architecture

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Modules 01-09 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Design an end-to-end production ML monitoring architecture
- Deploy a complete monitoring pipeline using Docker Compose
- Integrate data drift detection, performance tracking, alerting, and retraining
- Build operational runbooks for common ML monitoring scenarios
- Implement logging, tracing, and observability for ML systems

---

## Concepts

### Production Monitoring Architecture

A production-grade ML monitoring system integrates everything from Modules 01-09 into a cohesive pipeline:

```
+----------------+     +-------------------+     +------------------+
| Data Pipeline  |     | ML Model Service  |     | Monitoring       |
| (Feature Store)|---->| (FastAPI/Flask)   |---->| Pipeline         |
|                |     |                   |     |                  |
| - Feature      |     | - Prediction API  |     | - Data Drift     |
|   extraction   |     | - Batch scoring   |     | - Performance    |
| - Data quality |     | - A/B routing     |     | - Feature drift  |
|   checks       |     | - Logging         |     | - Concept drift  |
+--------+-------+     +---------+---------+     +--------+---------+
         |                       |                         |
         v                       v                         v
+--------+-------+     +---------+---------+     +---------+--------+
| Reference      |     | Prediction Store  |     | Metric Store     |
| Data Store     |     | (logs, outcomes)  |     | (Prometheus)     |
+----------------+     +-------------------+     +--------+---------+
                                                          |
                               +-------------------+     |
                               | Dashboard Layer   |<----+
                               | (Grafana)         |
                               +--------+----------+
                                        |
                               +--------v----------+
                               | Alert Manager     |
                               | (routing, dedup)  |
                               +--------+----------+
                                        |
                         +--------------+--------------+
                         |              |              |
                    +----v---+    +-----v----+   +-----v-----+
                    | Slack  |    | PagerDuty|   | Retraining|
                    |        |    |          |   | Pipeline  |
                    +--------+    +----------+   +-----------+
```

### The Complete Pipeline

#### Phase 1: Data Ingestion and Validation

```python
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_performance import ModelPerformanceTracker
from src.monitoring.alerting import AlertManager, AlertRule, AlertSeverity, AlertChannel
from src.retraining.trigger import RetrainingTrigger, TriggerCondition, TriggerReason

class ProductionMonitor:
    """
    End-to-end production monitoring pipeline.

    Integrates drift detection, performance tracking, alerting,
    and retraining triggers into a single orchestrated workflow.
    """

    def __init__(self, reference_data, model, config):
        # Initialize components
        self.drift_detector = DataDriftDetector(
            reference_data=reference_data,
            ks_threshold=config.get("ks_threshold", 0.05),
            psi_threshold=config.get("psi_threshold", 0.2),
        )

        self.performance_tracker = ModelPerformanceTracker(
            task_type=config.get("task_type", "classification"),
            degradation_threshold=config.get("degradation_threshold", 0.05),
        )

        self.alert_manager = AlertManager(
            alertmanager_url=config.get("alertmanager_url"),
            slack_webhook_url=config.get("slack_webhook_url"),
        )

        self.retraining_trigger = RetrainingTrigger(
            pipeline_url=config.get("pipeline_url"),
            validation_threshold=config.get("validation_threshold", 0.90),
        )

        self.model = model
        self._setup_alert_rules(config)
        self._setup_retraining_conditions(config)

    def _setup_alert_rules(self, config):
        """Configure default alert rules."""
        self.alert_manager.add_rule(AlertRule(
            name="data_drift_warning",
            metric="dataset_drift_share",
            condition="gt",
            threshold=0.3,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
            cooldown_minutes=30,
        ))
        self.alert_manager.add_rule(AlertRule(
            name="data_drift_critical",
            metric="dataset_drift_share",
            condition="gt",
            threshold=0.5,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.ALERTMANAGER],
            cooldown_minutes=10,
        ))

    def _setup_retraining_conditions(self, config):
        """Configure default retraining triggers."""
        self.retraining_trigger.add_condition(TriggerCondition(
            name="drift_retrain",
            reason=TriggerReason.DATA_DRIFT,
            metric="dataset_drift_share",
            condition="gt",
            threshold=0.5,
            min_samples=1000,
            cooldown_hours=24,
        ))

    def run_monitoring_cycle(self, current_data, y_true=None, y_pred=None, y_proba=None):
        """
        Execute a complete monitoring cycle.

        This is the main entry point called on each batch of production data.
        """
        results = {"timestamp": datetime.utcnow().isoformat()}

        # Step 1: Data Drift Detection
        drift_result = self.drift_detector.run_full_analysis(current_data)
        results["drift"] = drift_result.to_dict()

        # Step 2: Performance Tracking (if ground truth available)
        if y_true is not None and y_pred is not None:
            perf_alerts = self.performance_tracker.evaluate(y_true, y_pred, y_proba)
            results["performance"] = [a.to_dict() for a in perf_alerts]
        else:
            results["performance"] = None

        # Step 3: Alerting
        alerts = self.alert_manager.evaluate_drift_result(drift_result.to_dict())
        if results["performance"]:
            alerts.extend(
                self.alert_manager.evaluate_performance_alerts(results["performance"])
            )
        results["alerts_fired"] = len(alerts)

        # Step 4: Retraining Trigger
        retrain_run = self.retraining_trigger.evaluate_drift_result(
            drift_result.to_dict()
        )
        results["retraining_triggered"] = retrain_run is not None
        if retrain_run:
            results["retraining_run"] = retrain_run.to_dict()

        return results
```

#### Phase 2: Deployment with Docker Compose

The project's `docker-compose.yml` deploys the complete stack:

```bash
# Deploy the full monitoring stack
docker compose up -d

# Check all services
docker compose ps

# View logs
docker compose logs -f app
```

**Services deployed:**
| Service | Port | Purpose |
|---|---|---|
| `app` | 8000 | FastAPI monitoring API |
| `prometheus` | 9090 | Metric storage and alerting rules |
| `grafana` | 3000 | Visualization dashboards |
| `alertmanager` | 9093 | Alert routing and deduplication |
| `pushgateway` | 9091 | Push-based metric collection |

#### Phase 3: Operational Runbooks

##### Runbook: Data Drift Detected

```
1. CHECK: Open Grafana dashboard, identify drifted features
2. ASSESS: Are drifted features high-importance?
   - Yes -> Continue to step 3
   - No  -> Monitor for 24 hours, check if performance is affected
3. INVESTIGATE: Check upstream data pipeline
   - Any recent pipeline changes?
   - Any data source changes?
   - Any schema changes?
4. DECIDE:
   - Root cause identified and fixable -> Fix upstream
   - Distribution genuinely shifted -> Trigger retraining
   - Seasonal/expected -> Update reference data
5. ACT: Execute decision
6. VERIFY: Confirm metrics return to normal
7. DOCUMENT: Log incident and resolution
```

##### Runbook: Performance Degradation

```
1. CHECK: Confirm degradation is real (not a monitoring artifact)
   - Check sample sizes
   - Check for label delays
2. DIAGNOSE:
   - Data drift present? -> See drift runbook
   - No data drift? -> Likely concept drift
3. RESPOND:
   - If concept drift: Retrain with recent labeled data
   - If upstream issue: Fix data pipeline
   - If model bug: Rollback to previous version
4. VALIDATE: New model passes validation gate
5. DEPLOY: Use canary deployment (10% traffic)
6. MONITOR: Watch for 24-48 hours
7. SCALE: Increase to 100% if stable
```

### Logging and Observability

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Production-grade structured logging for ML monitoring."""

    def __init__(self, service_name="model-monitoring"):
        self.logger = logging.getLogger(service_name)
        self.service_name = service_name

    def log_prediction(self, request_id, features, prediction, confidence, latency_ms):
        self.logger.info(json.dumps({
            "event": "prediction",
            "request_id": request_id,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
        }))

    def log_drift_check(self, drift_result):
        self.logger.info(json.dumps({
            "event": "drift_check",
            "dataset_drifted": drift_result["is_dataset_drifted"],
            "drift_share": drift_result["drift_share"],
            "drifted_features": drift_result["drifted_features"],
            "total_features": drift_result["total_features"],
            "timestamp": datetime.utcnow().isoformat(),
        }))

    def log_retraining(self, run_info):
        self.logger.info(json.dumps({
            "event": "retraining",
            "run_id": run_info["run_id"],
            "reason": run_info["reason"],
            "status": run_info["status"],
            "timestamp": datetime.utcnow().isoformat(),
        }))
```

### Architecture Checklist for Production

| Category | Requirement | Status |
|---|---|---|
| **Data** | Reference data versioned and stored | |
| **Data** | Feature logging enabled | |
| **Data** | Prediction logging enabled | |
| **Monitoring** | Data drift detection automated | |
| **Monitoring** | Feature drift per-column tracking | |
| **Monitoring** | Performance tracking with ground truth | |
| **Monitoring** | Prediction distribution monitoring | |
| **Alerting** | Alert rules configured | |
| **Alerting** | Severity-based routing | |
| **Alerting** | Cooldowns and deduplication | |
| **Alerting** | Runbooks linked to alerts | |
| **Retraining** | Automated triggers configured | |
| **Retraining** | Validation gate before deployment | |
| **Retraining** | Rollback mechanism | |
| **Retraining** | A/B testing capability | |
| **Infra** | Prometheus metrics collection | |
| **Infra** | Grafana dashboards | |
| **Infra** | Structured logging | |
| **Infra** | Docker containerization | |
| **Infra** | Health checks | |

---

## Hands-On Lab

### Exercise 1: Deploy the Full Stack

**Goal:** Deploy the complete monitoring system.

```bash
# Clone and set up
cp .env.example .env

# Build and deploy
docker compose up -d --build

# Verify all services are healthy
docker compose ps

# Check the monitoring API
curl http://localhost:8000/health

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | python -m json.tool
```

### Exercise 2: Run a Complete Monitoring Cycle

**Goal:** Execute the full monitoring pipeline end-to-end.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up model and data
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

columns = [f"feature_{i}" for i in range(10)]
reference_df = pd.DataFrame(X_train[:5000], columns=columns)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize production monitor
config = {
    "task_type": "classification",
    "degradation_threshold": 0.05,
    "ks_threshold": 0.05,
    "psi_threshold": 0.2,
}

monitor = ProductionMonitor(reference_df, model, config)

# Set baseline
y_baseline_pred = model.predict(X_test)
y_baseline_proba = model.predict_proba(X_test)
monitor.performance_tracker.set_baseline(y_test, y_baseline_pred, y_baseline_proba)

# Simulate production with drift
X_drifted = X_test + np.random.normal(0.5, 0.3, X_test.shape)
current_df = pd.DataFrame(X_drifted[:1000], columns=columns)

y_pred = model.predict(X_drifted[:1000])
y_proba = model.predict_proba(X_drifted[:1000])

results = monitor.run_monitoring_cycle(
    current_data=current_df,
    y_true=y_test[:1000],
    y_pred=y_pred,
    y_proba=y_proba,
)

print("\nMonitoring Cycle Results:")
print(f"  Dataset drifted: {results['drift']['is_dataset_drifted']}")
print(f"  Drift share: {results['drift']['drift_share']:.2%}")
print(f"  Alerts fired: {results['alerts_fired']}")
print(f"  Retraining triggered: {results['retraining_triggered']}")
```

### Exercise 3: Build a Production Grafana Dashboard

**Goal:** Create a comprehensive Grafana dashboard.

1. Open Grafana at `http://localhost:3000`
2. Import or create a dashboard with these panels:

**Row 1: Overview**
- Model accuracy gauge
- Dataset drift share gauge
- Alerts fired (last 24h) counter
- Active model version

**Row 2: Drift Analysis**
- Drift scores by feature (time series)
- Drifted feature count (bar chart)
- Correlation drift distance (time series)

**Row 3: Performance**
- Accuracy, F1, AUC over time (time series)
- Prediction latency percentiles (histogram)
- Prediction volume (rate counter)

**Row 4: Retraining**
- Retraining events (annotations)
- Model version history (state timeline)
- Validation metrics per version (table)

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No health checks | Silent failures go unnoticed | Add `/health` endpoint with dependency checks |
| Monitoring system itself not monitored | Monitoring goes down silently | Use external uptime monitoring (e.g., Pingdom) |
| No structured logging | Cannot debug production issues | Use JSON structured logging with correlation IDs |
| Over-engineering from day one | Never ships | Start simple, iterate based on real incidents |
| Not testing the alerting pipeline | Alerts fail during real incidents | Regularly send test alerts end-to-end |

---

## Self-Check Questions

1. What are the five phases of a production monitoring pipeline?
2. How do you ensure the monitoring system itself is reliable?
3. What should a good operational runbook include?
4. How do you handle the case where ground truth is never available?
5. What is the role of structured logging in ML observability?

---

## You Know You Have Completed This Module When...

- [ ] You have deployed the full monitoring stack with Docker Compose
- [ ] You have run a complete monitoring cycle end-to-end
- [ ] You have built a production Grafana dashboard
- [ ] You have written operational runbooks for common scenarios
- [ ] Validation script passes: `bash modules/10-production-monitoring/validation/validate.sh`
- [ ] You can explain the entire monitoring architecture to a hiring manager

---

## Troubleshooting

### Common Issues

**Issue: Docker Compose services fail to start**
```bash
docker compose down -v  # Remove volumes and restart clean
docker compose up -d --build
docker compose logs     # Check for errors
```

**Issue: Prometheus not scraping metrics**
- Verify the app exposes `/metrics` endpoint
- Check `prometheus.yml` target configuration
- Ensure containers are on the same Docker network

**Issue: Grafana dashboard shows "No data"**
- Verify Prometheus datasource is configured correctly
- Check that metrics are being generated
- Ensure the time range is correct in Grafana

---

**Next: [Capstone Project ->](../../capstone/)**
