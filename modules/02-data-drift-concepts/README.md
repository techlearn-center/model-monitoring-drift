# Module 02: Data Drift Detection -- Concepts, Types, and Statistical Methods

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 01 completed, basic statistics knowledge |

---

## Learning Objectives

By the end of this module, you will be able to:

- Define data drift and explain its impact on production ML models
- Implement four core statistical methods for drift detection: PSI, KS test, Jensen-Shannon divergence, and Wasserstein distance
- Interpret drift test results and decide when to act
- Compare numerical and categorical drift detection approaches
- Choose the right drift detection method for your use case

---

## Concepts

### What Is Data Drift?

Data drift (also called **covariate shift**) occurs when the statistical distribution of production input data P_prod(X) diverges from the training distribution P_train(X). The model itself has not changed, but the data it receives no longer looks like what it was trained on.

```
Training Time:     P_train(X) --> Model --> Predictions (accurate)
Production (t+1):  P_prod(X)  --> Model --> Predictions (still ok)
Production (t+6):  P_drift(X) --> Model --> Predictions (degraded!)
                   ^^^^^^^^^
                   Distribution has shifted
```

### Types of Data Drift

| Drift Type | Description | Speed | Example |
|---|---|---|---|
| **Sudden Drift** | Abrupt distribution change | Instantaneous | System migration changes data format |
| **Gradual Drift** | Slow, progressive shift | Weeks to months | Customer demographics shift over time |
| **Incremental Drift** | Small step-wise changes | Periodic | Seasonal patterns (holiday spending) |
| **Recurring Drift** | Cyclical patterns | Periodic | Weather-dependent features |
| **Temporary Drift** | Short-lived anomaly | Brief | Marketing campaign spike |

### The Four Core Statistical Methods

#### 1. Population Stability Index (PSI)

PSI measures how much a distribution has shifted from a reference. Originally developed for credit scoring, it is one of the most widely used drift metrics in industry.

**Formula:**
```
PSI = SUM( (P_current(i) - P_reference(i)) * ln(P_current(i) / P_reference(i)) )
```

where i represents each bin of the discretized distributions.

**Interpretation:**
| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.1 | No significant shift | Continue monitoring |
| 0.1 - 0.2 | Moderate shift | Investigate, may need retraining |
| > 0.2 | Significant shift | Retrain model, investigate root cause |

**Implementation:**
```python
import numpy as np

def calculate_psi(reference, current, n_bins=10):
    """Calculate Population Stability Index."""
    eps = 1e-4

    # Create bins from reference distribution
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    # Count observations in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to proportions
    ref_pct = (ref_counts / ref_counts.sum()) + eps
    cur_pct = (cur_counts / cur_counts.sum()) + eps

    # Calculate PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi
```

**Strengths:** Simple, interpretable, well-established thresholds.
**Weaknesses:** Sensitive to bin count, does not capture distribution shape well.

#### 2. Kolmogorov-Smirnov (KS) Test

The KS test is a non-parametric test that measures the maximum distance between two cumulative distribution functions (CDFs). It produces both a test statistic and a p-value.

**How it works:**
```
KS statistic = max|F_reference(x) - F_current(x)|
```

The KS statistic D ranges from 0 (identical distributions) to 1 (completely different distributions). The p-value indicates the probability that the two samples come from the same distribution.

**Implementation:**
```python
from scipy import stats

def ks_drift_test(reference, current, alpha=0.05):
    """Two-sample Kolmogorov-Smirnov test for drift detection."""
    statistic, p_value = stats.ks_2samp(reference, current)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_drifted": p_value < alpha,
        "interpretation": (
            "Drift detected" if p_value < alpha
            else "No significant drift"
        )
    }
```

**Strengths:** Non-parametric, no binning needed, well-understood statistical test.
**Weaknesses:** Sensitive to sample size (large samples almost always detect "drift"), only works for continuous data.

#### 3. Jensen-Shannon (JS) Divergence

JS divergence is a symmetric, bounded version of Kullback-Leibler (KL) divergence. It measures the similarity between two probability distributions and always produces a value between 0 and 1.

**Formula:**
```
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

where M = 0.5 * (P + Q)  (mixture distribution)
```

**Implementation:**
```python
from scipy.spatial.distance import jensenshannon
import numpy as np

def js_divergence(reference, current, n_bins=10):
    """Calculate Jensen-Shannon divergence."""
    # Bin both distributions
    combined = np.concatenate([reference, current])
    bins = np.histogram_bin_edges(combined, bins=n_bins)

    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    # Normalize and add epsilon
    eps = 1e-10
    ref_hist = (ref_hist + eps) / (ref_hist + eps).sum()
    cur_hist = (cur_hist + eps) / (cur_hist + eps).sum()

    # JSD (scipy returns the square root, so we square it)
    jsd = jensenshannon(ref_hist, cur_hist) ** 2
    return jsd
```

**Strengths:** Symmetric, bounded [0,1], handles zero probabilities gracefully.
**Weaknesses:** Requires binning for continuous data, no standard threshold.

#### 4. Wasserstein Distance (Earth Mover's Distance)

The Wasserstein distance measures the minimum "cost" of transforming one distribution into another. Think of it as the minimum amount of "work" needed to rearrange one pile of dirt into another shape.

**Implementation:**
```python
from scipy.stats import wasserstein_distance

def wasserstein_drift(reference, current):
    """Calculate Wasserstein (Earth Mover's) Distance."""
    distance = wasserstein_distance(reference, current)
    return distance
```

**Strengths:** Intuitive interpretation, captures distribution shape, works well with continuous data.
**Weaknesses:** Scale-dependent (normalize features first), no universal threshold.

### Choosing the Right Method

| Scenario | Recommended Method | Why |
|---|---|---|
| Credit scoring, regulatory | PSI | Industry standard, well-defined thresholds |
| General numerical drift | KS Test | Non-parametric, provides p-value |
| Comparing probability distributions | Jensen-Shannon | Symmetric, bounded, handles zeros |
| Measuring distribution distance | Wasserstein | Captures shape differences, intuitive |
| Categorical features | Chi-squared test | Designed for categorical data |
| High-dimensional data | Maximum Mean Discrepancy (MMD) | Kernel-based, handles multivariate |

### Numerical vs. Categorical Drift

| Aspect | Numerical Features | Categorical Features |
|---|---|---|
| Methods | KS, PSI, Wasserstein, JS | Chi-squared, PSI, JS |
| Visualization | Histograms, KDE plots, CDFs | Bar charts, frequency tables |
| Challenges | Binning choices, scale sensitivity | Cardinality, unseen categories |
| Example | "age" distribution shifts from 25-40 to 18-35 | "payment_method" gains new category "crypto" |

---

## Hands-On Lab

### Prerequisites Check

```bash
python --version  # 3.10+
pip install evidently scipy scikit-learn pandas numpy matplotlib
```

### Exercise 1: Implement All Four Methods

**Goal:** Apply PSI, KS, JS, and Wasserstein to detect drift in a simulated dataset.

```python
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

np.random.seed(42)

# Reference distribution (training data)
reference = np.random.normal(loc=50, scale=10, size=5000)

# Scenario 1: No drift
current_no_drift = np.random.normal(loc=50, scale=10, size=1000)

# Scenario 2: Mean shift (gradual drift)
current_mean_shift = np.random.normal(loc=55, scale=10, size=1000)

# Scenario 3: Variance change
current_var_change = np.random.normal(loc=50, scale=20, size=1000)

# Scenario 4: Distribution type change (sudden drift)
current_type_change = np.random.exponential(scale=50, size=1000)

# Apply all four methods to each scenario
scenarios = {
    "No Drift": current_no_drift,
    "Mean Shift": current_mean_shift,
    "Variance Change": current_var_change,
    "Distribution Change": current_type_change,
}

for name, current in scenarios.items():
    ks_stat, ks_p = stats.ks_2samp(reference, current)
    psi = calculate_psi(reference, current)
    jsd = js_divergence(reference, current)
    wd = stats.wasserstein_distance(reference, current)

    print(f"\n{name}:")
    print(f"  KS: stat={ks_stat:.4f}, p={ks_p:.6f}, drifted={ks_p < 0.05}")
    print(f"  PSI: {psi:.4f}, drifted={psi > 0.2}")
    print(f"  JSD: {jsd:.4f}")
    print(f"  Wasserstein: {wd:.4f}")
```

### Exercise 2: Visualize Drift

**Goal:** Create visualizations that show distribution shifts.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, current) in zip(axes.flatten(), scenarios.items()):
    ax.hist(reference, bins=50, alpha=0.5, density=True, label="Reference")
    ax.hist(current, bins=50, alpha=0.5, density=True, label="Current")
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.savefig("drift_comparison.png", dpi=150)
plt.show()
```

### Exercise 3: Use the Project's DataDriftDetector

**Goal:** Use the `src/monitoring/data_drift.py` module.

```python
import pandas as pd
from src.monitoring.data_drift import DataDriftDetector

# Create reference and current DataFrames
reference_df = pd.DataFrame({
    "age": np.random.normal(35, 10, 5000),
    "income": np.random.lognormal(10, 1, 5000),
    "score": np.random.uniform(300, 850, 5000),
})

current_df = pd.DataFrame({
    "age": np.random.normal(28, 12, 1000),       # shifted younger
    "income": np.random.lognormal(10, 1, 1000),   # no change
    "score": np.random.uniform(300, 850, 1000),    # no change
})

detector = DataDriftDetector(reference_data=reference_df)
result = detector.run_full_analysis(current_data=current_df)

print(f"Dataset drifted: {result.is_dataset_drifted}")
print(f"Drifted features: {result.drifted_features}/{result.total_features}")
for col_result in result.column_results:
    print(f"  {col_result.feature} ({col_result.method}): "
          f"stat={col_result.statistic:.4f}, drifted={col_result.is_drifted}")
```

---

## Starter Files

Check `lab/starter/` for:
- `drift_methods.py` -- Skeleton implementations to complete
- `drift_scenarios.py` -- Pre-built drift scenarios to test against
- `visualize_drift.py` -- Visualization templates

## Solution Files

If you get stuck, `lab/solution/` contains:
- Complete implementations of all four methods
- Visualization code with annotated plots
- Comparison analysis across methods

> **Important:** Try to complete the exercises yourself first!

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not normalizing features before Wasserstein | Large-scale features dominate | Standardize features before computing distance |
| Using too few bins for PSI | PSI values are unreliable | Use 10-20 bins, test sensitivity to bin count |
| Ignoring sample size effects on KS test | Everything flags as "drifted" | Use effect size alongside p-value, or adjust alpha |
| Applying numerical tests to categorical data | Nonsensical results | Use Chi-squared or PSI with category-based bins |
| Setting thresholds without validation | Too many false positives or missed drift | Calibrate thresholds on historical data |

---

## Self-Check Questions

1. What is the mathematical definition of PSI, and why are the thresholds 0.1 and 0.2 commonly used?
2. Why is the KS test problematic with very large sample sizes?
3. How does Jensen-Shannon divergence differ from Kullback-Leibler divergence?
4. When would you choose Wasserstein distance over PSI?
5. How would you handle categorical features with unseen categories in production?

---

## You Know You Have Completed This Module When...

- [ ] You can implement PSI, KS test, JS divergence, and Wasserstein distance from scratch
- [ ] You can interpret the output of each method and explain what it means
- [ ] You know which method to choose for different scenarios
- [ ] Validation script passes: `bash modules/02-data-drift-concepts/validation/validate.sh`
- [ ] You can explain these methods to a colleague in a code review

---

## Troubleshooting

### Common Issues

**Issue: scipy.stats.ks_2samp returns p-value of 0.0**
- This happens with very large samples. The KS test becomes overly sensitive.
- Solution: Use a subsample or switch to an effect-size metric like Wasserstein.

**Issue: PSI returns negative values**
- Check for empty bins or zero-count issues.
- Make sure you add epsilon to prevent log(0).

**Issue: Jensen-Shannon returns NaN**
- Ensure distributions are normalized and non-negative.
- Add a small epsilon before computing.

---

**Next: [Module 03 -- Evidently AI Setup and Dashboards ->](../03-evidently-ai-setup/)**
