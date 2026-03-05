# Module 05: Model Performance Tracking

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Track classification and regression model metrics over time
- Detect performance degradation using threshold-based and trend-based methods
- Build rolling window analysis for performance monitoring
- Set up Prometheus gauges and counters for real-time performance tracking
- Implement baseline comparison and degradation alerting

---

## Concepts

### Why Track Model Performance?

Data drift tells you the **input distribution has changed**, but it does not tell you whether the model is still making good predictions. Performance tracking directly measures **how well the model is doing its job**.

```
Data Drift Detected     =/=>    Performance Degraded
(Input changed)                 (Predictions wrong)

No Data Drift Detected  =/=>    Performance Stable
(Input unchanged)               (Concept may have drifted)
```

You need both signals: drift detection warns you early, and performance tracking confirms the impact.

### Classification Metrics Deep Dive

| Metric | Formula | Best For | Limitation |
|---|---|---|---|
| **Accuracy** | (TP+TN) / Total | Balanced classes | Misleading on imbalanced data |
| **Precision** | TP / (TP+FP) | When false positives are costly | Ignores false negatives |
| **Recall** | TP / (TP+FN) | When false negatives are costly | Ignores false positives |
| **F1 Score** | 2 * (P*R) / (P+R) | Balanced precision-recall tradeoff | No single metric fits all |
| **AUC-ROC** | Area under ROC curve | Ranking quality, threshold-independent | Can be optimistic on imbalanced data |
| **Log Loss** | -mean(y*log(p)) | Probability calibration | Sensitive to confident wrong predictions |

### Regression Metrics Deep Dive

| Metric | Formula | Best For | Limitation |
|---|---|---|---|
| **MSE** | mean((y - y_hat)^2) | Penalizing large errors | Sensitive to outliers, scale-dependent |
| **MAE** | mean(\|y - y_hat\|) | Robust to outliers | Does not penalize large errors enough |
| **RMSE** | sqrt(MSE) | Same unit as target | Same issues as MSE |
| **R-squared** | 1 - SS_res/SS_tot | Proportion of variance explained | Can be misleading with nonlinear data |
| **MAPE** | mean(\|y - y_hat\| / y) | Percentage-based comparison | Undefined when y = 0 |

### Performance Degradation Detection

#### Threshold-Based Detection

Compare current performance against a fixed baseline:

```
degradation = (baseline_value - current_value) / baseline_value

if degradation > threshold:
    alert("Performance degraded")
```

Example: If baseline accuracy = 0.95 and current accuracy = 0.89:
```
degradation = (0.95 - 0.89) / 0.95 = 0.063 = 6.3%
With threshold = 5%, this triggers an alert.
```

#### Trend-Based Detection

Monitor the slope of performance over a rolling window:

```
Performance
    |
0.95|  *  *
    |       *  *
0.90|            *  *
    |                 *  *
0.85|                      *
    +-------------------------> Time
         Negative slope = degradation trend
```

#### Statistical Process Control

Use control charts to detect out-of-control performance:

```
    UCL (Upper Control Limit) -------- +3 sigma
    |
    |    *   *    *   *
    |  *   *   *    *   *
    Mean ---------------------- center line
    |    *   *    *
    |  *   *
    |
    LCL (Lower Control Limit) -------- -3 sigma
```

### The Ground Truth Problem

A major challenge in production monitoring is that **ground truth (actual labels) often arrives with a delay** or never arrives at all.

| Scenario | Ground Truth Delay | Monitoring Strategy |
|---|---|---|
| Fraud detection | Days to weeks | Proxy metrics + data drift |
| Product recommendation | Minutes (click/no-click) | Near-real-time performance |
| Medical diagnosis | Months | Data drift + expert review |
| Ad click prediction | Seconds | Real-time performance tracking |
| Loan default prediction | Months to years | Heavily rely on data drift |

When ground truth is delayed, rely on:
1. **Prediction distribution monitoring** -- are predictions shifting?
2. **Data drift detection** -- has the input changed?
3. **Proxy metrics** -- partial signals (e.g., customer complaints, refund rates)
4. **Confidence monitoring** -- is the model becoming less confident?

---

## Hands-On Lab

### Exercise 1: Set Up Performance Tracking

**Goal:** Use the project's `ModelPerformanceTracker` class.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.monitoring.model_performance import ModelPerformanceTracker

# Create and train a model
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=12,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Set up tracker with 5% degradation threshold
tracker = ModelPerformanceTracker(
    task_type="classification",
    degradation_threshold=0.05,
)

# Establish baseline
y_val_pred = model.predict(X_test)
y_val_proba = model.predict_proba(X_test)
baseline = tracker.set_baseline(y_test, y_val_pred, y_val_proba)
print(f"Baseline: accuracy={baseline.accuracy:.4f}, "
      f"f1={baseline.f1:.4f}, auc={baseline.auc_roc:.4f}")

# Simulate production batches with increasing degradation
for batch in range(10):
    noise = np.random.normal(0, 0.1 * (batch + 1), X_test.shape)
    X_noisy = X_test + noise

    y_pred = model.predict(X_noisy)
    y_proba = model.predict_proba(X_noisy)
    alerts = tracker.evaluate(y_test, y_pred, y_proba)

    degraded = [a for a in alerts if a.is_degraded]
    if degraded:
        print(f"\nBatch {batch + 1}: DEGRADATION DETECTED")
        for alert in degraded:
            print(f"  {alert.metric}: {alert.baseline_value:.4f} -> "
                  f"{alert.current_value:.4f} ({alert.degradation:.1%} drop)")
    else:
        snapshot = tracker.history[-1]
        print(f"Batch {batch + 1}: OK (accuracy={snapshot.accuracy:.4f})")
```

### Exercise 2: Rolling Metrics and Trend Analysis

**Goal:** Analyze performance trends over time.

```python
import matplotlib.pyplot as plt

# Get rolling metrics DataFrame
df = tracker.get_rolling_metrics()
print(df[["accuracy", "f1", "auc_roc", "sample_size"]])

# Plot performance over time
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric in zip(axes, ["accuracy", "f1", "auc_roc"]):
    values = df[metric].dropna()
    ax.plot(range(len(values)), values, marker="o")
    ax.axhline(
        y=getattr(baseline, metric), color="r",
        linestyle="--", label="Baseline"
    )
    ax.set_title(metric.upper())
    ax.set_xlabel("Batch")
    ax.legend()

plt.tight_layout()
plt.savefig("reports/performance_trend.png", dpi=150)

# Check trend direction
for metric in ["accuracy", "f1", "auc_roc"]:
    trend = tracker.get_trend(metric)
    print(f"{metric}: direction={trend['direction']}, slope={trend['slope']:.6f}")
```

### Exercise 3: Regression Performance Tracking

**Goal:** Track regression model performance with MSE, MAE, and R-squared.

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

X_reg, y_reg = make_regression(
    n_samples=5000, n_features=10, noise=10, random_state=42
)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_reg, y_reg, test_size=0.3
)

reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_r_train, y_r_train)

reg_tracker = ModelPerformanceTracker(
    task_type="regression",
    degradation_threshold=0.10,
)

y_r_pred = reg_model.predict(X_r_test)
reg_baseline = reg_tracker.set_baseline(y_r_test, y_r_pred)
print(f"Regression baseline: MSE={reg_baseline.mse:.4f}, "
      f"MAE={reg_baseline.mae:.4f}, R2={reg_baseline.r2:.4f}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using accuracy on imbalanced data | Model looks good but fails on minority class | Use F1, AUC, or precision-recall curves |
| Setting thresholds too tight | Constant false alarms | Calibrate thresholds on historical variability |
| Ignoring delayed ground truth | Performance appears to jump or drop | Account for label delay in monitoring pipeline |
| Not tracking prediction distribution | Miss concept drift without ground truth | Monitor prediction confidence and distribution |
| Comparing batches of different sizes | Noisy metrics | Use fixed batch sizes or weighted statistics |

---

## Self-Check Questions

1. Why is tracking accuracy alone insufficient for monitoring model performance?
2. How do you handle the ground truth delay problem in production monitoring?
3. What is the difference between threshold-based and trend-based degradation detection?
4. When would you use MAE over MSE for regression monitoring?
5. How does a sliding window size affect the sensitivity of trend detection?

---

## You Know You Have Completed This Module When...

- [ ] You can track classification and regression metrics over time
- [ ] You understand how to set meaningful degradation thresholds
- [ ] You can detect performance trends using rolling window analysis
- [ ] Validation script passes: `bash modules/05-model-performance-tracking/validation/validate.sh`
- [ ] You can explain the ground truth delay problem and mitigation strategies

---

## Troubleshooting

### Common Issues

**Issue: AUC-ROC cannot be computed**
- Ensure you pass prediction probabilities, not just class labels
- Binary classification needs `y_proba[:, 1]`, not the full probability matrix

**Issue: Metrics are NaN**
- Check for empty batches or batches with a single class
- Ensure ground truth labels match prediction format

**Issue: Too many degradation alerts**
- Increase the degradation threshold
- Use trend-based detection instead of per-batch threshold comparison

---

**Next: [Module 06 -- Feature Drift Detection ->](../06-feature-drift-detection/)**
