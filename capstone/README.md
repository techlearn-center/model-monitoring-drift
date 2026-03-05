# Capstone Project: End-to-End ML Model Monitoring and Drift Detection System

## Overview

This capstone project combines everything you learned across all 10 modules into a single, production-grade ML monitoring system. You will build a complete pipeline that detects data drift, tracks model performance, fires alerts, and triggers automated retraining -- all deployed with Docker and observable through Grafana dashboards.

This is the project you will showcase to hiring managers and discuss in technical interviews.

---

## The Challenge

Build a production-ready model monitoring system for a classification model that demonstrates:

### Tier 1: Foundation (Modules 01-03)
- [ ] Train a classification model on a dataset of your choice (e.g., credit scoring, fraud detection, churn prediction)
- [ ] Store reference data (training distribution) for drift comparison
- [ ] Generate Evidently AI drift reports (HTML and JSON)
- [ ] Run Evidently test suites with pass/fail criteria

### Tier 2: Detection Engine (Modules 04-06)
- [ ] Implement PSI, KS test, Jensen-Shannon, and Wasserstein drift detection
- [ ] Track classification metrics over time (accuracy, F1, AUC, precision, recall)
- [ ] Monitor per-feature drift with importance-weighted scoring
- [ ] Detect correlation drift between features

### Tier 3: Alerting and Response (Modules 07-09)
- [ ] Monitor prediction distribution for concept drift signals
- [ ] Configure Prometheus metrics and Grafana dashboards
- [ ] Set up alert rules with severity levels, routing, and cooldowns
- [ ] Implement automated retraining triggers with validation gates
- [ ] Build A/B testing capability for model comparison

### Tier 4: Production Deployment (Module 10)
- [ ] Containerize the application with Docker
- [ ] Deploy the full stack with Docker Compose (app + Prometheus + Grafana + Alertmanager)
- [ ] Implement health checks and structured logging
- [ ] Write operational runbooks for drift and degradation scenarios
- [ ] Create a comprehensive Grafana monitoring dashboard

---

## Architecture

Design your solution following this reference architecture:

```
+------------------+     +-------------------+     +------------------+
| Data Pipeline    |     | Monitoring API    |     | Alert Pipeline   |
|                  |     | (FastAPI)         |     |                  |
| - Load reference |     | POST /analyze     |     | - Prometheus     |
| - Load current   |---->| POST /predict     |---->| - Alertmanager   |
| - Validate data  |     | GET  /metrics     |     | - Slack/Webhook  |
|                  |     | GET  /health      |     |                  |
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                         |
         v                         v                         v
+--------+---------+     +---------+---------+     +--------+---------+
| Reference Store  |     | Drift Detector    |     | Grafana          |
| (CSV/Parquet)    |     | Performance Track |     | (Dashboards)     |
|                  |     | Alerting Engine   |     |                  |
+------------------+     | Retrain Trigger   |     +------------------+
                          +-------------------+
```

Your system should be:
- **Reliable** -- Handles failures gracefully with health checks and retries
- **Observable** -- Every component produces structured logs and Prometheus metrics
- **Automated** -- Drift detection, alerting, and retraining triggers run without human intervention
- **Documented** -- Runbooks, architecture diagrams, and API documentation

---

## Getting Started

```bash
# 1. Review the requirements
cat capstone/requirements.md

# 2. Start with the starter files
ls capstone/starter/

# 3. Copy environment configuration
cp .env.example .env

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start building!
# Use src/monitoring/ and src/retraining/ as your foundation

# 6. Deploy with Docker
docker compose up -d --build

# 7. Validate your work
bash capstone/validation/validate.sh
```

---

## Implementation Guide

### Step 1: Choose Your Dataset

Pick a dataset that demonstrates clear drift scenarios:

| Dataset | Type | Drift Opportunity |
|---|---|---|
| Credit card fraud | Binary classification | Transaction pattern changes |
| Customer churn | Binary classification | Demographic shifts |
| Income prediction | Binary classification | Economic condition changes |
| Wine quality | Multi-class | Seasonal grape variations |
| Loan default | Binary classification | Policy and market changes |

### Step 2: Train Your Model

```python
# Example starter code
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv("data/your_dataset.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and reference data
joblib.dump(model, "models/production_model.pkl")
X_train.to_csv("data/reference.csv", index=False)
```

### Step 3: Build the Monitoring Pipeline

Use the project's `src/monitoring/` modules to build your pipeline:

```python
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_performance import ModelPerformanceTracker
from src.monitoring.alerting import AlertManager
from src.retraining.trigger import RetrainingTrigger
```

### Step 4: Deploy and Test

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

---

## Evaluation Criteria

| Criteria | Weight | What We Look For |
|---|---|---|
| **Functionality** | 30% | Does drift detection, alerting, and retraining work correctly? |
| **Architecture** | 20% | Clean separation of concerns, modular design |
| **Monitoring Coverage** | 15% | Data drift + performance + feature drift + alerting |
| **Automation** | 15% | How much runs without human intervention? |
| **Documentation** | 10% | Runbooks, architecture diagram, API docs |
| **Code Quality** | 10% | Clean code, type hints, docstrings, error handling |

---

## Showcasing to Hiring Managers

When you complete this capstone:

1. **Fork this repo** to your personal GitHub
2. **Add your solution** with detailed commit messages showing your thought process
3. **Record a 3-5 minute demo video** walking through:
   - The architecture and design decisions
   - A live demonstration of drift detection triggering alerts
   - The Grafana dashboard and how to read it
   - The automated retraining pipeline
4. **Prepare to discuss:**
   - Why you chose specific drift detection methods
   - How you set thresholds and why
   - How this system would scale in a real production environment
   - What you would add with more time (feature store, model registry, CI/CD)

See [docs/portfolio-guide.md](../docs/portfolio-guide.md) for detailed guidance.

---

## Solution

The `solution/` directory contains a reference implementation. Try to complete the capstone yourself first -- that is what builds real skills and interview confidence.

```bash
# Only look at this after giving it your best attempt
ls capstone/solution/
```

---

## Validation

```bash
# Run the capstone validation
bash capstone/validation/validate.sh

# Expected output: all checks PASS
```

The validation script checks:
- All required components are present
- Docker services start successfully
- API endpoints respond correctly
- Drift detection produces valid results
- Alert rules are configured
- Grafana dashboard is accessible
