# Module 09: Retraining Triggers and Pipelines

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Design automated retraining triggers based on drift and performance signals
- Build a retraining pipeline with data selection, training, validation, and deployment
- Implement model validation gates to prevent bad models from reaching production
- Set up A/B testing infrastructure for comparing retrained models against production
- Handle rollback scenarios when a retrained model underperforms

---

## Concepts

### When to Retrain

Retraining is not always the right response to drift. Use this decision framework:

```
Drift or degradation detected
         |
         v
Is performance actually degraded?
    |              |
   YES             NO
    |              |
    v              v
Is there enough   Monitor closely,
new data?         may be false alarm
    |
   YES
    |
    v
Trigger retraining pipeline
    |
    v
Validate new model > baseline?
    |              |
   YES             NO
    |              |
    v              v
Deploy new        Keep old model,
model             investigate
```

### Retraining Trigger Types

| Trigger Type | Signal | When to Use | Risk |
|---|---|---|---|
| **Drift-Based** | Data drift score > threshold | Input distribution changed significantly | May retrain unnecessarily |
| **Performance-Based** | Accuracy/F1 below threshold | Direct evidence of degradation | Requires ground truth |
| **Schedule-Based** | Fixed interval (daily, weekly) | Predictable data evolution | Wastes compute if no drift |
| **Volume-Based** | N new labeled samples collected | Enough data for meaningful retraining | May not correlate with drift |
| **Manual** | Engineer judgment | Complex situations | Slow response time |

### The Retraining Pipeline

```
+--------+    +----------+    +--------+    +----------+    +--------+
| Data   |--->| Feature  |--->| Model  |--->| Validate |--->| Deploy |
| Select |    | Engineer |    | Train  |    | (gate)   |    |        |
+--------+    +----------+    +--------+    +----------+    +--------+
     ^                                           |               |
     |                                      PASS |          FAIL |
     |                                           v               v
     |                                      New model       Keep old
     |                                      ready           model
     +-------------------------------------------<---------Rollback
```

#### Step 1: Data Selection

Choose training data carefully:

```python
def select_training_data(
    historical_data,
    recent_data,
    strategy="combined",
    recent_weight=0.7,
    max_samples=50000,
):
    """
    Select data for retraining.

    Strategies:
    - "recent_only": Only use recent data (adapts fastest, forgets past)
    - "all_historical": Use all data (most stable, slowest to adapt)
    - "combined": Weighted combination (balanced approach)
    - "sliding_window": Fixed-size window of most recent data
    """
    if strategy == "recent_only":
        return recent_data.sample(min(len(recent_data), max_samples))

    elif strategy == "all_historical":
        combined = pd.concat([historical_data, recent_data])
        return combined.sample(min(len(combined), max_samples))

    elif strategy == "combined":
        n_recent = int(max_samples * recent_weight)
        n_historical = max_samples - n_recent

        recent_sample = recent_data.sample(min(len(recent_data), n_recent))
        hist_sample = historical_data.sample(min(len(historical_data), n_historical))

        return pd.concat([hist_sample, recent_sample]).sample(frac=1)

    elif strategy == "sliding_window":
        combined = pd.concat([historical_data, recent_data])
        return combined.tail(max_samples)
```

#### Step 2: Model Validation Gate

Never deploy a model that has not passed validation:

```python
def validation_gate(
    new_model,
    old_model,
    X_val,
    y_val,
    min_improvement=-0.01,  # Allow up to 1% drop
    required_metrics=None,
):
    """
    Validation gate: decide whether the new model replaces the old one.

    Args:
        new_model: Retrained model
        old_model: Currently deployed model
        X_val: Validation features
        y_val: Validation labels
        min_improvement: Minimum required improvement (negative = allow small drop)
        required_metrics: Dict of {metric: min_value}

    Returns:
        Dict with decision and metrics comparison
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    new_preds = new_model.predict(X_val)
    old_preds = old_model.predict(X_val)

    metrics = {
        "accuracy": (accuracy_score(y_val, new_preds), accuracy_score(y_val, old_preds)),
        "f1": (f1_score(y_val, new_preds, average="weighted"),
               f1_score(y_val, old_preds, average="weighted")),
    }

    # Check if new model meets minimum requirements
    if required_metrics:
        for metric, min_val in required_metrics.items():
            new_val = metrics[metric][0]
            if new_val < min_val:
                return {
                    "decision": "REJECT",
                    "reason": f"{metric}={new_val:.4f} below minimum {min_val}",
                    "metrics": metrics,
                }

    # Check if new model is better than old model
    primary_metric = "f1"
    new_val, old_val = metrics[primary_metric]
    improvement = new_val - old_val

    if improvement >= min_improvement:
        return {
            "decision": "APPROVE",
            "reason": f"{primary_metric} improved by {improvement:.4f}",
            "metrics": metrics,
        }
    else:
        return {
            "decision": "REJECT",
            "reason": f"{primary_metric} dropped by {abs(improvement):.4f}",
            "metrics": metrics,
        }
```

### A/B Testing for Retrained Models

Before fully deploying a retrained model, run it alongside the production model:

```
                    +----> Model A (current) ---> 90% traffic
Incoming traffic ---+
                    +----> Model B (retrained) --> 10% traffic
                                                      |
                                            Compare performance
                                            on real traffic
```

**A/B Test Design:**
| Parameter | Recommendation |
|---|---|
| Traffic split | Start with 5-10% to new model |
| Duration | At least 1 week or 10,000 predictions |
| Primary metric | Business metric (revenue, conversion) |
| Statistical test | Two-proportion z-test or Bayesian |
| Rollback criteria | New model performs worse by > 2% |

```python
import numpy as np
from scipy import stats

def ab_test_significance(
    control_successes, control_total,
    treatment_successes, treatment_total,
    alpha=0.05,
):
    """
    Two-proportion z-test for A/B testing retrained models.

    Args:
        control_successes: Correct predictions from Model A
        control_total: Total predictions from Model A
        treatment_successes: Correct predictions from Model B
        treatment_total: Total predictions from Model B
        alpha: Significance level

    Returns:
        Dict with test results and recommendation
    """
    p1 = control_successes / control_total
    p2 = treatment_successes / treatment_total

    # Pooled proportion
    p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treatment_total))

    # Z-statistic
    z_stat = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "control_rate": p1,
        "treatment_rate": p2,
        "difference": p2 - p1,
        "z_statistic": z_stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "recommendation": (
            "Deploy treatment model" if p2 > p1 and p_value < alpha
            else "Keep control model" if p1 >= p2 and p_value < alpha
            else "Continue testing (not yet significant)"
        ),
    }
```

### Rollback Strategy

Always have a plan to revert:

| Approach | Speed | Risk |
|---|---|---|
| **Blue-Green Deployment** | Instant rollback | Requires 2x infrastructure |
| **Canary Release** | Gradual rollback | Catches issues early |
| **Feature Flag** | Instant toggle | Requires feature flag system |
| **Model Registry** | Fast rollback | Requires versioned model store |

---

## Hands-On Lab

### Exercise 1: Set Up Retraining Triggers

**Goal:** Configure the `RetrainingTrigger` class from the project.

```python
from src.retraining.trigger import (
    RetrainingTrigger, TriggerCondition, TriggerReason
)

trigger = RetrainingTrigger(
    validation_threshold=0.90,
    auto_rollback=True,
    current_model_version="v1.0.0",
)

# Drift-based trigger
trigger.add_condition(TriggerCondition(
    name="high_data_drift",
    reason=TriggerReason.DATA_DRIFT,
    metric="dataset_drift_share",
    condition="gt",
    threshold=0.5,
    min_samples=1000,
    cooldown_hours=24,
))

# Performance-based trigger
trigger.add_condition(TriggerCondition(
    name="accuracy_drop",
    reason=TriggerReason.PERFORMANCE_DEGRADATION,
    metric="perf_accuracy",
    condition="lt",
    threshold=0.85,
    min_samples=500,
    cooldown_hours=12,
))

# Simulate evaluation
run = trigger.evaluate(
    metric_name="dataset_drift_share",
    value=0.65,
    sample_count=5000,
)

if run:
    print(f"Retraining triggered!")
    print(f"  Run ID: {run.run_id}")
    print(f"  Reason: {run.reason.value}")
    print(f"  Status: {run.status.value}")
```

### Exercise 2: Build a Validation Gate

**Goal:** Implement model validation before deployment.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simulate old and new models
X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# "Old" model (fewer trees, worse performance)
old_model = RandomForestClassifier(n_estimators=10, random_state=42)
old_model.fit(X_train, y_train)

# "New" retrained model (more trees, better performance)
new_model = RandomForestClassifier(n_estimators=100, random_state=42)
new_model.fit(X_train, y_train)

result = validation_gate(
    new_model, old_model, X_val, y_val,
    min_improvement=-0.01,
    required_metrics={"accuracy": 0.85},
)

print(f"Decision: {result['decision']}")
print(f"Reason: {result['reason']}")
for metric, (new_val, old_val) in result["metrics"].items():
    print(f"  {metric}: old={old_val:.4f}, new={new_val:.4f}, delta={new_val-old_val:+.4f}")
```

### Exercise 3: A/B Test Simulation

**Goal:** Run a simulated A/B test between two model versions.

```python
# Simulate A/B test data
np.random.seed(42)

# Model A: 88% accuracy on 10,000 predictions
control_total = 10000
control_correct = int(control_total * 0.88)

# Model B: 90% accuracy on 1,000 predictions
treatment_total = 1000
treatment_correct = int(treatment_total * 0.90)

result = ab_test_significance(
    control_correct, control_total,
    treatment_correct, treatment_total,
)

print("A/B Test Results:")
for key, value in result.items():
    print(f"  {key}: {value}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No validation gate | Bad models reach production | Always validate before deploying |
| Retraining too frequently | Wasted compute, unstable models | Use cooldown periods and minimum sample counts |
| Using only recent data | Model forgets important patterns | Combine recent and historical data |
| No rollback plan | Stuck with bad model in production | Implement blue-green or canary deployment |
| A/B test too short | Inconclusive results | Run until statistically significant |
| Not versioning models | Cannot reproduce or rollback | Store every model version in a registry |

---

## Self-Check Questions

1. What are the five types of retraining triggers, and when would you use each?
2. Why is a validation gate essential before deploying a retrained model?
3. How do you choose between "recent only" and "combined" data selection strategies?
4. What statistical test would you use to compare two models in an A/B test?
5. How would you design a rollback strategy for a production ML system?

---

## You Know You Have Completed This Module When...

- [ ] You can configure automated retraining triggers based on drift signals
- [ ] You have implemented a model validation gate
- [ ] You understand A/B testing methodology for model comparison
- [ ] Validation script passes: `bash modules/09-retraining-triggers/validation/validate.sh`
- [ ] You can design a complete retraining pipeline with rollback

---

## Troubleshooting

### Common Issues

**Issue: Retraining trigger fires too often**
- Increase cooldown_hours
- Increase min_samples requirement
- Raise the drift threshold

**Issue: Validation gate always rejects new model**
- Check that new model is trained on enough data
- Lower min_improvement threshold
- Ensure validation data is representative

**Issue: A/B test never reaches significance**
- Need more traffic or longer duration
- The difference between models may be too small to detect
- Consider using Bayesian A/B testing for faster decisions

---

**Next: [Module 10 -- Production Monitoring System ->](../10-production-monitoring/)**
