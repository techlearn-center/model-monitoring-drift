# Module 06: Feature Drift Detection

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 05 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Monitor individual features for drift using per-feature statistical tests
- Detect changes in feature correlations and multivariate relationships
- Prioritize drift alerts based on feature importance to the model
- Track feature quality metrics (missing values, outliers, cardinality)
- Build a feature monitoring dashboard that highlights actionable drift

---

## Concepts

### Why Monitor Features Individually?

Dataset-level drift gives you a single "drifted or not" signal. But to **diagnose and fix** drift, you need to know **which specific features** have shifted and **how much** they have changed.

```
Dataset drift detected!
   |
   +-- Which features?
       |
       +-- age: PSI = 0.42 (DRIFTED - shifted younger)
       +-- income: PSI = 0.03 (stable)
       +-- region: PSI = 0.08 (stable)
       +-- payment_method: PSI = 0.31 (DRIFTED - new category "crypto")
       |
       +-- Root cause: Marketing campaign targeted younger audience
           with crypto payment option
```

### Feature Monitoring Dimensions

For each feature, track these dimensions:

| Dimension | What to Monitor | Example Check |
|---|---|---|
| **Distribution** | Statistical shape changes | KS test, PSI, histograms |
| **Central Tendency** | Mean, median shifts | Mean within N sigma of reference |
| **Spread** | Variance, range changes | Levene's test, IQR comparison |
| **Missing Values** | Null rate changes | Missing rate vs. reference |
| **Outliers** | Extreme value frequency | Outlier rate vs. reference |
| **Cardinality** | Unique value count (categorical) | New categories, dropped categories |
| **Correlation** | Relationship between features | Correlation matrix drift |

### Feature Importance-Weighted Monitoring

Not all features are equal. A drift in a feature with high model importance is far more likely to affect predictions than drift in a low-importance feature.

```python
import numpy as np

def weighted_drift_score(drift_scores, feature_importances):
    """
    Compute importance-weighted drift score.

    Args:
        drift_scores: Dict of {feature: drift_score}
        feature_importances: Dict of {feature: importance}

    Returns:
        Weighted drift score (higher = more concerning)
    """
    total_importance = sum(feature_importances.values())
    weighted_score = 0.0

    for feature, drift in drift_scores.items():
        importance = feature_importances.get(feature, 0)
        weight = importance / total_importance
        weighted_score += drift * weight

    return weighted_score
```

**Priority matrix:**
| | High Drift | Low Drift |
|---|---|---|
| **High Importance** | CRITICAL -- act immediately | Monitor closely |
| **Low Importance** | Investigate cause | Acceptable, log only |

### Correlation Drift

Feature correlations can change even when individual features appear stable. This is called **multivariate drift** or **correlation drift**.

```python
import numpy as np
import pandas as pd

def correlation_drift(reference_df, current_df, threshold=0.15):
    """
    Detect changes in feature correlation structure.

    Compares correlation matrices using Frobenius norm.
    """
    ref_corr = reference_df.select_dtypes(include=[np.number]).corr()
    cur_corr = current_df.select_dtypes(include=[np.number]).corr()

    # Frobenius norm of the difference
    diff = ref_corr - cur_corr
    frobenius_norm = np.sqrt((diff ** 2).sum().sum())

    # Normalize by matrix size
    n = len(ref_corr)
    normalized_distance = frobenius_norm / n

    # Find the most changed correlations
    max_change_idx = np.unravel_index(
        np.abs(diff.values).argmax(), diff.shape
    )
    max_change_features = (
        diff.index[max_change_idx[0]],
        diff.columns[max_change_idx[1]],
    )
    max_change_value = diff.iloc[max_change_idx[0], max_change_idx[1]]

    return {
        "frobenius_norm": frobenius_norm,
        "normalized_distance": normalized_distance,
        "is_drifted": normalized_distance > threshold,
        "max_change_features": max_change_features,
        "max_change_value": max_change_value,
    }
```

### Feature Quality Monitoring

Beyond drift, monitor data quality signals that indicate upstream pipeline issues:

```python
def feature_quality_check(reference_series, current_series):
    """Check data quality metrics for a single feature."""
    checks = {}

    # Missing value rate
    ref_missing_rate = reference_series.isna().mean()
    cur_missing_rate = current_series.isna().mean()
    checks["missing_rate_change"] = cur_missing_rate - ref_missing_rate

    # Numeric-specific checks
    if np.issubdtype(current_series.dtype, np.number):
        ref_clean = reference_series.dropna()
        cur_clean = current_series.dropna()

        # Range check
        checks["min_in_range"] = cur_clean.min() >= ref_clean.min() * 0.5
        checks["max_in_range"] = cur_clean.max() <= ref_clean.max() * 2.0

        # Outlier rate (beyond 3 sigma)
        ref_mean, ref_std = ref_clean.mean(), ref_clean.std()
        ref_outlier_rate = (
            (ref_clean < ref_mean - 3 * ref_std) |
            (ref_clean > ref_mean + 3 * ref_std)
        ).mean()
        cur_outlier_rate = (
            (cur_clean < ref_mean - 3 * ref_std) |
            (cur_clean > ref_mean + 3 * ref_std)
        ).mean()
        checks["outlier_rate_change"] = cur_outlier_rate - ref_outlier_rate

    # Categorical-specific checks
    if current_series.dtype == "object" or current_series.dtype.name == "category":
        ref_categories = set(reference_series.dropna().unique())
        cur_categories = set(current_series.dropna().unique())
        checks["new_categories"] = list(cur_categories - ref_categories)
        checks["missing_categories"] = list(ref_categories - cur_categories)

    return checks
```

---

## Hands-On Lab

### Exercise 1: Per-Feature Drift Analysis

**Goal:** Run drift detection on each feature and rank by severity.

```python
import numpy as np
import pandas as pd
from src.monitoring.data_drift import DataDriftDetector

# Generate reference data
np.random.seed(42)
reference_df = pd.DataFrame({
    "age": np.random.normal(35, 10, 5000),
    "income": np.random.lognormal(10, 1, 5000),
    "score": np.random.uniform(300, 850, 5000),
    "tenure": np.random.exponential(5, 5000),
    "balance": np.random.normal(50000, 15000, 5000),
})

# Current data: drift in some features only
current_df = pd.DataFrame({
    "age": np.random.normal(28, 12, 1000),        # DRIFTED
    "income": np.random.lognormal(10, 1, 1000),    # Stable
    "score": np.random.uniform(300, 850, 1000),    # Stable
    "tenure": np.random.exponential(3, 1000),      # DRIFTED
    "balance": np.random.normal(50000, 15000, 1000),  # Stable
})

# Feature importances from a trained model
feature_importances = {
    "age": 0.30,
    "income": 0.25,
    "score": 0.20,
    "tenure": 0.15,
    "balance": 0.10,
}

# Detect per-feature drift
detector = DataDriftDetector(reference_data=reference_df)
result = detector.run_full_analysis(current_data=current_df)

# Rank features by importance-weighted drift
print("\nFeature Drift Report (ranked by importance):")
print("-" * 60)

for feat in sorted(feature_importances, key=feature_importances.get, reverse=True):
    feat_results = [r for r in result.column_results if r.feature == feat and r.method == "ks_test"]
    if feat_results:
        r = feat_results[0]
        importance = feature_importances[feat]
        priority = "CRITICAL" if r.is_drifted and importance > 0.2 else \
                   "WARNING" if r.is_drifted else "OK"
        print(f"  [{priority:8}] {feat:12} importance={importance:.2f} "
              f"KS={r.statistic:.4f} drifted={r.is_drifted}")
```

### Exercise 2: Correlation Drift Detection

**Goal:** Detect changes in feature correlation structure.

```python
import matplotlib.pyplot as plt

# Add correlation to reference data
reference_df["age_income"] = reference_df["age"] * 1000 + np.random.normal(0, 5000, 5000)
reference_df["income"] = reference_df["age_income"]  # Strong correlation

# Break correlation in current data
current_df["age_income"] = np.random.lognormal(10, 1, 1000)  # No correlation
current_df["income"] = current_df["age_income"]

# Check correlation drift
corr_result = correlation_drift(reference_df, current_df)
print(f"Correlation drift detected: {corr_result['is_drifted']}")
print(f"Normalized distance: {corr_result['normalized_distance']:.4f}")
print(f"Largest change: {corr_result['max_change_features']} "
      f"(delta = {corr_result['max_change_value']:.4f})")
```

### Exercise 3: Feature Quality Dashboard

**Goal:** Build a comprehensive feature quality check.

```python
# Inject quality issues
current_with_issues = current_df.copy()
current_with_issues.loc[:50, "age"] = np.nan  # Add missing values
current_with_issues.loc[0, "balance"] = 9999999  # Add outlier

for feature in reference_df.columns:
    if feature in current_with_issues.columns:
        quality = feature_quality_check(
            reference_df[feature], current_with_issues[feature]
        )
        print(f"\n{feature}:")
        for check, value in quality.items():
            print(f"  {check}: {value}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Treating all features equally | Alert fatigue from low-importance drift | Weight drift alerts by feature importance |
| Ignoring correlation changes | Multivariate drift goes undetected | Monitor correlation matrix stability |
| Not tracking new categories | Model encounters unknown categorical values | Alert on new category appearance |
| Monitoring too many features | Alert overload | Focus on top-k important features |
| Static reference data | Reference becomes outdated | Periodically refresh reference with validated data |

---

## Self-Check Questions

1. Why is per-feature drift analysis more actionable than dataset-level drift?
2. How does feature importance affect your monitoring strategy?
3. What is correlation drift and why can it occur even when individual features are stable?
4. How would you handle a new category appearing in a categorical feature?
5. What feature quality metrics should you track beyond distribution drift?

---

## You Know You Have Completed This Module When...

- [ ] You can run per-feature drift analysis and rank features by severity
- [ ] You understand importance-weighted drift scoring
- [ ] You can detect correlation drift using matrix comparison
- [ ] Validation script passes: `bash modules/06-feature-drift-detection/validation/validate.sh`
- [ ] You can build a feature quality dashboard

---

## Troubleshooting

### Common Issues

**Issue: Correlation matrix has NaN values**
- Drop columns with all missing values before computing correlation
- Ensure you filter to numeric columns only

**Issue: Feature importance not available**
- Use model-agnostic importance (e.g., permutation importance)
- Fall back to equal weighting if no model is available

---

**Next: [Module 07 -- Prediction and Concept Drift ->](../07-prediction-drift/)**
