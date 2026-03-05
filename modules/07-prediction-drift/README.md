# Module 07: Prediction Drift and Concept Drift

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Distinguish between data drift, prediction drift, and concept drift
- Detect concept drift (posterior drift) when ground truth is available
- Identify gradual, sudden, incremental, and recurring concept drift patterns
- Monitor prediction distributions as a proxy for concept drift
- Implement concept drift detection with and without ground truth labels

---

## Concepts

### The Three Pillars of ML Drift

```
+------------------+     +--------------------+     +-------------------+
|   DATA DRIFT     |     | PREDICTION DRIFT   |     |  CONCEPT DRIFT    |
|   P(X) changes   |     | P(Y_hat) changes   |     |  P(Y|X) changes   |
+------------------+     +--------------------+     +-------------------+
| Input features   |     | Model outputs      |     | True relationship |
| shift in         |     | distribution       |     | between X and Y   |
| distribution     |     | changes            |     | changes           |
+--------+---------+     +---------+----------+     +--------+----------+
         |                         |                          |
    Detectable             Detectable without          Requires ground
    without labels         ground truth                truth (delayed)
```

### What Is Concept Drift?

Concept drift occurs when the **true relationship** between input features and the target variable changes over time. The learned function f(X) -> Y no longer represents reality.

**Mathematical definition:**
- At training time: P_train(Y|X) defines the true mapping
- At production time: P_prod(Y|X) differs from P_train(Y|X)
- The model still applies P_train(Y|X), producing incorrect predictions

**Example -- fraud detection:**
- Training: Transactions > $10,000 from new accounts = high fraud risk
- After 6 months: Fraudsters adapt, now use many small transactions ($50-200)
- The concept of "what constitutes fraud" has drifted

### Types of Concept Drift

| Type | Pattern | Speed | Example | Detection Difficulty |
|---|---|---|---|---|
| **Sudden** | Abrupt change | Instantaneous | Regulatory change, system migration | Easiest -- clear before/after |
| **Gradual** | Slow transition | Weeks to months | Consumer behavior evolution | Medium -- hard to pinpoint start |
| **Incremental** | Step-wise | Periodic steps | Quarterly policy updates | Medium -- looks like noise initially |
| **Recurring** | Cyclical | Seasonal | Holiday shopping patterns | Easy once pattern is known |

```
Sudden:          Gradual:         Incremental:      Recurring:
    |                 /                ___|               /\
    |____        ____/            ___|              /\  /  \
         |      /             ___|                 /  \/    \
         |_____/           __|                    /          \
```

### Prediction Drift (Without Ground Truth)

When ground truth is not available, monitor the **distribution of model predictions** as a proxy signal.

```python
import numpy as np
from scipy import stats

def detect_prediction_drift(
    reference_predictions,
    current_predictions,
    alpha=0.05,
):
    """
    Detect drift in model prediction distributions.

    This works WITHOUT ground truth labels by comparing
    the distribution of predictions over time.
    """
    results = {}

    # Distribution of predicted classes (classification)
    if np.issubdtype(type(reference_predictions[0]), np.integer):
        ref_dist = np.bincount(reference_predictions) / len(reference_predictions)
        cur_dist = np.bincount(current_predictions) / len(current_predictions)

        # Pad to same length
        max_len = max(len(ref_dist), len(cur_dist))
        ref_dist = np.pad(ref_dist, (0, max_len - len(ref_dist)))
        cur_dist = np.pad(cur_dist, (0, max_len - len(cur_dist)))

        # Chi-squared test on prediction distribution
        from scipy.stats import chi2_contingency
        contingency = np.array([
            ref_dist * len(reference_predictions),
            cur_dist * len(current_predictions),
        ])
        chi2, p_value, _, _ = chi2_contingency(contingency + 1e-10)
        results["prediction_distribution"] = {
            "chi2": chi2, "p_value": p_value,
            "drifted": p_value < alpha,
        }

    # Distribution of prediction probabilities
    else:
        ks_stat, ks_p = stats.ks_2samp(reference_predictions, current_predictions)
        results["prediction_distribution"] = {
            "ks_statistic": ks_stat, "p_value": ks_p,
            "drifted": ks_p < alpha,
        }

    # Prediction rate changes
    ref_mean = np.mean(reference_predictions)
    cur_mean = np.mean(current_predictions)
    results["mean_prediction_shift"] = {
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "absolute_change": abs(cur_mean - ref_mean),
        "relative_change": abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-10),
    }

    return results
```

### Concept Drift Detection (With Ground Truth)

When ground truth becomes available (even with delay), use these methods:

#### 1. Page-Hinkley Test
Detects changes in the mean of a sequence. Good for detecting gradual drift.

```python
class PageHinkleyDetector:
    """
    Page-Hinkley test for concept drift detection.

    Monitors a running sum of differences from the mean.
    Triggers when the sum exceeds a threshold.
    """

    def __init__(self, delta=0.005, threshold=50, alpha=0.9999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.running_mean = 0.0
        self.min_sum = float("inf")

    def update(self, value):
        """Add a new observation and check for drift."""
        self.n += 1
        self.running_mean = (
            self.running_mean * (self.n - 1) + value
        ) / self.n
        self.sum += value - self.running_mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)

        drift_detected = (self.sum - self.min_sum) > self.threshold

        return {
            "drift_detected": drift_detected,
            "test_statistic": self.sum - self.min_sum,
            "running_mean": self.running_mean,
            "n_observations": self.n,
        }
```

#### 2. ADWIN (Adaptive Windowing)
Maintains a variable-length window and detects distributional changes.

```python
class SimpleADWIN:
    """
    Simplified ADWIN-style detector.

    Compares recent observations against older ones using
    a sliding window with adaptive size.
    """

    def __init__(self, max_window=1000, delta=0.01):
        self.max_window = max_window
        self.delta = delta
        self.window = []

    def update(self, value):
        """Add observation and check for drift."""
        self.window.append(value)

        # Trim to max window
        if len(self.window) > self.max_window:
            self.window = self.window[-self.max_window:]

        if len(self.window) < 20:
            return {"drift_detected": False, "n_observations": len(self.window)}

        # Compare first half vs second half
        mid = len(self.window) // 2
        first_half = np.array(self.window[:mid])
        second_half = np.array(self.window[mid:])

        mean_diff = abs(first_half.mean() - second_half.mean())
        pooled_var = (first_half.var() + second_half.var()) / 2
        bound = np.sqrt(2 * pooled_var * np.log(2 / self.delta) / mid)

        drift_detected = mean_diff > bound

        if drift_detected:
            # Shrink window to recent data
            self.window = self.window[mid:]

        return {
            "drift_detected": drift_detected,
            "mean_diff": mean_diff,
            "bound": bound,
            "window_size": len(self.window),
        }
```

#### 3. Error Rate Monitoring

The most direct approach: track model error rate over time and detect when it increases.

```python
def error_rate_drift(errors, window_size=100, threshold=0.10):
    """
    Monitor error rate over a sliding window.

    Args:
        errors: List of 0/1 (correct/incorrect predictions)
        window_size: Size of sliding window
        threshold: Maximum acceptable error rate increase
    """
    if len(errors) < window_size * 2:
        return {"drift_detected": False, "reason": "insufficient data"}

    baseline_errors = errors[:window_size]
    recent_errors = errors[-window_size:]

    baseline_rate = np.mean(baseline_errors)
    recent_rate = np.mean(recent_errors)

    increase = recent_rate - baseline_rate

    return {
        "drift_detected": increase > threshold,
        "baseline_error_rate": baseline_rate,
        "current_error_rate": recent_rate,
        "error_rate_increase": increase,
    }
```

---

## Hands-On Lab

### Exercise 1: Simulate and Detect Concept Drift

**Goal:** Create synthetic concept drift scenarios and detect them.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Phase 1: Original concept (y = 1 if x1 + x2 > 0)
X_phase1 = np.random.normal(0, 1, (2000, 2))
y_phase1 = (X_phase1[:, 0] + X_phase1[:, 1] > 0).astype(int)

# Phase 2: Concept drift (y = 1 if x1 - x2 > 0)
X_phase2 = np.random.normal(0, 1, (2000, 2))
y_phase2 = (X_phase2[:, 0] - X_phase2[:, 1] > 0).astype(int)

# Train on Phase 1
model = LogisticRegression()
model.fit(X_phase1[:1000], y_phase1[:1000])

# Evaluate on Phase 1 (should be good)
acc_phase1 = accuracy_score(y_phase1[1000:], model.predict(X_phase1[1000:]))
print(f"Phase 1 accuracy: {acc_phase1:.4f}")

# Evaluate on Phase 2 (concept has drifted)
acc_phase2 = accuracy_score(y_phase2, model.predict(X_phase2))
print(f"Phase 2 accuracy: {acc_phase2:.4f}")
print(f"Accuracy drop: {acc_phase1 - acc_phase2:.4f}")

# Use Page-Hinkley to detect the drift point
ph = PageHinkleyDetector(delta=0.005, threshold=10)
all_errors = []

for i in range(1000):
    pred = model.predict(X_phase1[1000 + i:1001 + i])[0]
    error = int(pred != y_phase1[1000 + i])
    all_errors.append(error)
    result = ph.update(error)

for i in range(1000):
    pred = model.predict(X_phase2[i:i+1])[0]
    error = int(pred != y_phase2[i])
    all_errors.append(error)
    result = ph.update(error)
    if result["drift_detected"]:
        print(f"Drift detected at observation {1000 + i}!")
        break
```

### Exercise 2: Prediction Drift Without Labels

**Goal:** Detect drift using only prediction distributions.

```python
# Reference predictions (Phase 1)
ref_preds = model.predict_proba(X_phase1[1000:])[:, 1]

# Current predictions (Phase 2 -- concept has drifted)
cur_preds = model.predict_proba(X_phase2[:1000])[:, 1]

drift_result = detect_prediction_drift(ref_preds, cur_preds)
print("\nPrediction drift analysis:")
for key, value in drift_result.items():
    print(f"  {key}: {value}")
```

### Exercise 3: Gradual Concept Drift Simulation

**Goal:** Detect gradual concept drift using ADWIN.

```python
adwin = SimpleADWIN(max_window=500, delta=0.01)

# Simulate gradual drift: slowly change the decision boundary
for t in range(2000):
    # Gradually shift the concept
    drift_amount = t / 2000.0  # 0 to 1
    x = np.random.normal(0, 1, 2)

    # Interpolate between concepts
    concept1 = (x[0] + x[1] > 0)
    concept2 = (x[0] - x[1] > 0)
    true_label = concept1 if np.random.random() > drift_amount else concept2

    pred = model.predict(x.reshape(1, -1))[0]
    error = int(pred != true_label)

    result = adwin.update(error)
    if result["drift_detected"]:
        print(f"ADWIN drift detected at t={t}: "
              f"mean_diff={result['mean_diff']:.4f}, "
              f"window_size={result['window_size']}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Confusing data drift with concept drift | Retrain on same concept, no improvement | Check if P(Y\|X) has changed, not just P(X) |
| Waiting for ground truth to detect any drift | Late detection | Use prediction distribution as early warning |
| Setting Page-Hinkley threshold too low | Constant false alarms | Calibrate on stable period first |
| Ignoring seasonal concept drift | "Drift" detected every quarter | Model seasonal patterns explicitly |
| Not distinguishing drift types | Wrong remediation strategy | Diagnose if drift is sudden, gradual, or recurring |

---

## Self-Check Questions

1. What is the mathematical difference between data drift P(X) and concept drift P(Y|X)?
2. How can you detect concept drift when ground truth labels are delayed by weeks?
3. What is the difference between sudden and gradual concept drift, and how does detection strategy differ?
4. Why is prediction distribution monitoring useful even without ground truth?
5. When would you use Page-Hinkley vs. ADWIN for drift detection?

---

## You Know You Have Completed This Module When...

- [ ] You can distinguish between data drift, prediction drift, and concept drift
- [ ] You have implemented Page-Hinkley and ADWIN detectors
- [ ] You can detect concept drift with and without ground truth
- [ ] Validation script passes: `bash modules/07-prediction-drift/validation/validate.sh`
- [ ] You can simulate and detect gradual, sudden, and recurring drift patterns

---

## Troubleshooting

### Common Issues

**Issue: Page-Hinkley never triggers**
- Decrease the threshold parameter
- Increase delta for more sensitivity
- Check that your error signal has sufficient variation

**Issue: ADWIN triggers too frequently**
- Increase the delta parameter (makes it less sensitive)
- Increase the minimum window size
- Check for natural variability in your error signal

---

**Next: [Module 08 -- Automated Alerts and Actions ->](../08-automated-alerts/)**
