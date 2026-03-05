# Module 01: ML Monitoring Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Python 3.10+, Docker installed, basic ML knowledge |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain why ML models degrade in production and the business impact of unmonitored models
- Distinguish between data drift, concept drift, and model performance degradation
- Identify the key metrics and signals that indicate a model is failing
- Set up a basic monitoring pipeline with logging and metrics collection
- Articulate the ML monitoring lifecycle from deployment to retraining

---

## Concepts

### Why Do ML Models Degrade in Production?

Unlike traditional software, ML models are **fundamentally tied to the data they were trained on**. When the real world changes, trained models become stale. This phenomenon is called **model decay** or **model staleness**.

Consider a fraud detection model trained on 2023 transaction patterns. By mid-2024, new payment methods, economic conditions, and fraud tactics have emerged. The model's accuracy silently erodes because:

1. **The world changes** -- customer behavior, market conditions, and external factors shift over time
2. **Data distributions shift** -- the statistical properties of incoming data no longer match training data
3. **Feedback loops** -- the model's own predictions alter the system it monitors
4. **Upstream data changes** -- data pipelines, feature engineering, or third-party data sources change without notice
5. **Seasonal patterns** -- cyclical trends that were not captured in training data

**Real-world example:** A major ride-sharing company reported that their demand prediction models lost 5-10% accuracy every quarter without retraining. An e-commerce recommendation engine saw click-through rates drop by 30% within 6 months of deployment because product catalog and user preferences evolved.

### The Three Types of Model Degradation

| Type | What Changes | Detection Method | Example |
|---|---|---|---|
| **Data Drift** | Input feature distributions P(X) | Statistical tests on input data | Customer age distribution shifts younger |
| **Concept Drift** | Relationship between features and target P(Y\|X) | Performance monitoring with ground truth | What constitutes "spam" email evolves |
| **Performance Degradation** | Model accuracy, F1, AUC metrics | Metric tracking dashboards | Accuracy drops from 0.95 to 0.82 |

### The ML Monitoring Stack

A production monitoring system has four layers:

```
+----------------------------------------------------------+
|  Layer 4: ALERTING & ACTION                              |
|  PagerDuty, Slack, automated retraining triggers         |
+----------------------------------------------------------+
|  Layer 3: DASHBOARDS & VISUALIZATION                     |
|  Grafana dashboards, Evidently reports                   |
+----------------------------------------------------------+
|  Layer 2: METRICS & DETECTION                            |
|  Prometheus metrics, statistical drift tests             |
+----------------------------------------------------------+
|  Layer 1: DATA COLLECTION & LOGGING                      |
|  Feature logging, prediction logging, ground truth       |
+----------------------------------------------------------+
```

### Key Terminology

| Term | Definition |
|---|---|
| **Data Drift** | Change in the distribution of input features P(X) between training and production |
| **Concept Drift** | Change in the relationship between inputs and outputs P(Y\|X) |
| **Model Staleness** | Gradual degradation of model performance over time |
| **Reference Dataset** | The baseline dataset (usually training data) that represents "normal" |
| **Current Dataset** | Recent production data being compared against the reference |
| **Ground Truth** | The actual correct labels/values, often available with a delay |
| **Feature Store** | A centralized repository for storing and serving ML features |
| **Shadow Mode** | Running a new model alongside production without serving its predictions |
| **Canary Deployment** | Gradually routing traffic to a new model to detect issues early |

### The Cost of Not Monitoring

| Scenario | Impact |
|---|---|
| Undetected data drift for 3 months | Model serves incorrect predictions to millions of users |
| No performance tracking | Revenue loss from degraded recommendations |
| No alerting pipeline | Engineering team discovers issues from customer complaints |
| No retraining trigger | Manual ad-hoc retraining, inconsistent model quality |

---

## Hands-On Lab

### Prerequisites Check

Before starting, verify your environment:

```bash
# Check Python version (3.10+ required)
python --version

# Check Docker is running
docker --version
docker compose version

# Check you have the project cloned
ls modules/01-monitoring-fundamentals/
```

### Exercise 1: Understanding Model Degradation

**Goal:** Simulate model degradation by introducing distribution shift.

**Step 1:** Install the project dependencies
```bash
pip install -r requirements.txt
```

**Step 2:** Create a simple model and observe degradation
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification

# Generate training data
X_train, y_train = make_classification(
    n_samples=5000, n_features=10, n_informative=6,
    random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on "normal" test data
X_test, y_test = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    random_state=42
)
print(f"Normal accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

# Simulate data drift -- shift feature distributions
X_drifted = X_test + np.random.normal(loc=2.0, scale=0.5, size=X_test.shape)
print(f"Drifted accuracy: {accuracy_score(y_test, model.predict(X_drifted)):.4f}")
```

**What you should see:** The accuracy drops significantly on drifted data compared to normal test data.

### Exercise 2: Setting Up Basic Monitoring

**Goal:** Log predictions and set up metric collection.

```python
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_monitor")

def predict_with_monitoring(model, X, feature_names=None):
    """Make predictions while logging monitoring signals."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Log prediction distribution
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    confidence_mean = probabilities.max(axis=1).mean()

    logger.info(
        "Batch prediction: n=%d, pred_mean=%.4f, pred_std=%.4f, "
        "avg_confidence=%.4f, timestamp=%s",
        len(X), pred_mean, pred_std, confidence_mean,
        datetime.utcnow().isoformat()
    )

    # Alert on low confidence
    low_confidence = (probabilities.max(axis=1) < 0.6).sum()
    if low_confidence > len(X) * 0.1:
        logger.warning(
            "HIGH LOW-CONFIDENCE RATE: %d/%d predictions (%.1f%%) "
            "have confidence < 0.6",
            low_confidence, len(X), low_confidence / len(X) * 100
        )

    return predictions
```

### Exercise 3: Connecting Metrics to Prometheus

**Goal:** Expose monitoring metrics via Prometheus client.

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Define metrics
prediction_counter = Counter(
    'model_predictions_total', 'Total predictions made'
)
prediction_latency = Histogram(
    'model_prediction_latency_seconds', 'Prediction latency'
)
accuracy_gauge = Gauge(
    'model_accuracy_current', 'Current model accuracy'
)

# Start Prometheus metrics server
start_http_server(8001)
print("Prometheus metrics available at http://localhost:8001/metrics")
```

Run the validation to check your setup:
```bash
bash modules/01-monitoring-fundamentals/validation/validate.sh
```

---

## Starter Files

Check `lab/starter/` for:
- `simulate_drift.py` -- Script to generate drifted datasets
- `basic_monitor.py` -- Skeleton monitoring class to complete
- `docker-compose.yml` -- Local Prometheus + Grafana setup

## Solution Files

If you get stuck, `lab/solution/` contains:
- Complete monitoring implementation
- Working Prometheus integration
- Expected output examples

> **Important:** Try to complete the exercises yourself first! Looking at solutions too early reduces learning.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not logging prediction distributions | Cannot detect drift without ground truth | Always log input features and prediction statistics |
| Monitoring only accuracy | Miss upstream data issues | Monitor data quality, distributions, and latency too |
| Using training data as the only reference | Drift detected where none exists | Use a validated hold-out set as reference |
| No alerting thresholds | Dashboard exists but nobody checks it | Set up automated alerts from day one |
| Ignoring feature importance changes | Model silently relies on wrong features | Track feature importance over time |

---

## Self-Check Questions

Test your understanding before moving on:

1. What are the three types of model degradation, and how does each one manifest?
2. Why can a model have high accuracy on test data but perform poorly in production?
3. What is the difference between data drift and concept drift? Give a real-world example of each.
4. Why is ground truth often delayed in production, and how does this affect monitoring?
5. What are the four layers of an ML monitoring stack, and what does each layer provide?

---

## You Know You Have Completed This Module When...

- [ ] You can explain three reasons why ML models degrade in production
- [ ] You have simulated model degradation using distribution shift
- [ ] You have implemented basic prediction logging with monitoring signals
- [ ] You understand the difference between data drift and concept drift
- [ ] Validation script passes: `bash modules/01-monitoring-fundamentals/validation/validate.sh`
- [ ] You can describe the ML monitoring stack to a teammate

---

## Troubleshooting

### Common Issues

**Issue: Python package installation fails**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Issue: Prometheus metrics endpoint not accessible**
```bash
# Check if port 8001 is available
lsof -i :8001
# Use a different port if needed
```

**Issue: Docker containers not starting**
```bash
docker compose logs <service-name>  # Check logs
docker compose down && docker compose up -d  # Restart
```

---

**Next: [Module 02 -- Data Drift Concepts and Types ->](../02-data-drift-concepts/)**
