# Module 04: Statistical Tests for Drift Detection

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed, statistics fundamentals |

---

## Learning Objectives

By the end of this module, you will be able to:

- Apply parametric and non-parametric statistical tests for drift detection
- Implement the Kolmogorov-Smirnov test, Chi-squared test, and Population Stability Index
- Understand hypothesis testing fundamentals: null hypothesis, p-values, significance levels
- Handle multiple testing correction with Bonferroni and Benjamini-Hochberg methods
- Choose the right test for numerical vs. categorical vs. multivariate data

---

## Concepts

### Hypothesis Testing for Drift

Drift detection is fundamentally a **hypothesis testing problem**:

- **Null Hypothesis (H0):** The reference and current distributions are the same -- no drift has occurred
- **Alternative Hypothesis (H1):** The distributions differ -- drift has occurred

We reject H0 (declare drift) when the test statistic exceeds a critical value, or equivalently, when the p-value falls below our significance level (alpha).

```
                   p-value < alpha
                         |
              +----------+----------+
              |                     |
         p < 0.05               p >= 0.05
     "Drift detected"      "No significant drift"
     (Reject H0)            (Fail to reject H0)
```

**Critical consideration:** A small p-value does not mean the drift is practically significant. With large sample sizes, even trivial differences become statistically significant. Always pair statistical tests with effect size measures.

### Comprehensive Test Reference

#### For Numerical Features

| Test | Type | What It Measures | Best For |
|---|---|---|---|
| **Kolmogorov-Smirnov** | Non-parametric | Max CDF distance | General distribution comparison |
| **Anderson-Darling** | Non-parametric | Weighted CDF distance | Sensitive to distribution tails |
| **Mann-Whitney U** | Non-parametric | Rank-based comparison | Ordinal data, non-normal distributions |
| **Welch's t-test** | Parametric | Mean difference | Comparing means of normal distributions |
| **Levene's test** | Parametric | Variance equality | Detecting variance changes |
| **Cramér-von Mises** | Non-parametric | Integrated CDF distance | Sensitive to shape differences |
| **Wasserstein** | Distance | Earth mover's distance | Continuous shape comparison |
| **PSI** | Information | Distribution stability | Industry standard, binned comparison |

#### For Categorical Features

| Test | What It Measures | Best For |
|---|---|---|
| **Chi-squared** | Independence of distributions | Comparing category frequencies |
| **Fisher's exact** | Exact independence test | Small sample sizes |
| **PSI (categorical)** | Stability of category proportions | Monitoring category distribution shifts |
| **Jensen-Shannon** | Distribution similarity | Comparing probability distributions |

### Deep Dive: Kolmogorov-Smirnov Test

The two-sample KS test compares the empirical CDFs of two samples:

```python
from scipy import stats
import numpy as np

def detailed_ks_test(reference, current, alpha=0.05):
    """
    Detailed KS test with interpretation.

    The KS statistic D measures the maximum vertical distance
    between the two empirical CDFs.
    """
    stat, p_value = stats.ks_2samp(reference, current)

    # Critical value approximation for two-sample test
    n1, n2 = len(reference), len(current)
    c_alpha = np.sqrt(-0.5 * np.log(alpha / 2))
    critical_value = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))

    return {
        "statistic": stat,
        "p_value": p_value,
        "critical_value": critical_value,
        "is_drifted": p_value < alpha,
        "effect_size": stat,  # KS stat IS the effect size
        "interpretation": interpret_ks(stat, p_value, alpha),
        "sample_sizes": {"reference": n1, "current": n2},
    }

def interpret_ks(stat, p_value, alpha):
    """Human-readable interpretation of KS test results."""
    if p_value >= alpha:
        return f"No significant drift detected (p={p_value:.4f} >= {alpha})"
    elif stat < 0.1:
        return f"Statistically significant but small drift (D={stat:.4f})"
    elif stat < 0.2:
        return f"Moderate drift detected (D={stat:.4f})"
    else:
        return f"Large drift detected (D={stat:.4f})"
```

### Deep Dive: Chi-Squared Test for Categorical Features

```python
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def chi_squared_drift_test(reference_series, current_series, alpha=0.05):
    """
    Chi-squared test for categorical drift detection.

    Compares the frequency distribution of categories between
    reference and current data.
    """
    # Get all categories from both datasets
    all_categories = set(reference_series.unique()) | set(current_series.unique())

    # Build frequency table
    ref_counts = reference_series.value_counts()
    cur_counts = current_series.value_counts()

    # Align categories
    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
    cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]

    # Create contingency table
    contingency = np.array([ref_aligned, cur_aligned])

    # Remove columns where both are zero
    nonzero_mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero_mask]

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Effect size: Cramér's V
    n = contingency.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "cramers_v": cramers_v,
        "is_drifted": p_value < alpha,
        "new_categories": list(
            set(current_series.unique()) - set(reference_series.unique())
        ),
    }
```

### Multiple Testing Correction

When testing drift on many features simultaneously, the probability of false positives increases. With 100 features at alpha=0.05, you expect approximately 5 false drift detections.

**Bonferroni correction:** Divide alpha by the number of tests.
```python
alpha_corrected = 0.05 / n_features  # Very conservative
```

**Benjamini-Hochberg (FDR):** Controls the false discovery rate.
```python
from statsmodels.stats.multitest import multipletests

p_values = [test_result["p_value"] for test_result in results]
reject, corrected_p, _, _ = multipletests(p_values, method="fdr_bh")
```

---

## Hands-On Lab

### Exercise 1: Multi-Method Comparison

**Goal:** Apply all statistical tests to the same dataset and compare results.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

reference = np.random.normal(0, 1, 5000)
slight_drift = np.random.normal(0.2, 1, 1000)
moderate_drift = np.random.normal(0.5, 1.2, 1000)
severe_drift = np.random.normal(1.5, 2.0, 1000)

drifts = {
    "Slight (mean +0.2)": slight_drift,
    "Moderate (mean +0.5, std 1.2)": moderate_drift,
    "Severe (mean +1.5, std 2.0)": severe_drift,
}

for name, current in drifts.items():
    print(f"\n{'='*50}")
    print(f"Scenario: {name}")

    # KS Test
    ks_stat, ks_p = stats.ks_2samp(reference, current)
    print(f"  KS Test:         D={ks_stat:.4f}, p={ks_p:.6f}")

    # Anderson-Darling
    ad_stat, _, ad_p = stats.anderson_ksamp([reference, current])
    print(f"  Anderson-Darling: A={ad_stat:.4f}, p={ad_p:.6f}")

    # Mann-Whitney U
    mw_stat, mw_p = stats.mannwhitneyu(reference, current)
    print(f"  Mann-Whitney U:  U={mw_stat:.0f}, p={mw_p:.6f}")

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(reference, current, equal_var=False)
    print(f"  Welch's t-test:  t={t_stat:.4f}, p={t_p:.6f}")
```

### Exercise 2: Multiple Testing Correction

**Goal:** Apply Bonferroni and Benjamini-Hochberg correction to a multi-feature scenario.

```python
from statsmodels.stats.multitest import multipletests

# Simulate 50 features, only 5 truly drifted
n_features = 50
n_truly_drifted = 5
p_values = []

for i in range(n_features):
    ref = np.random.normal(0, 1, 5000)
    if i < n_truly_drifted:
        cur = np.random.normal(0.8, 1, 1000)  # Drifted
    else:
        cur = np.random.normal(0, 1, 1000)     # Not drifted

    _, p = stats.ks_2samp(ref, cur)
    p_values.append(p)

# Without correction
detected_raw = sum(1 for p in p_values if p < 0.05)

# Bonferroni
reject_bonf, p_bonf, _, _ = multipletests(p_values, method="bonferroni")

# Benjamini-Hochberg
reject_bh, p_bh, _, _ = multipletests(p_values, method="fdr_bh")

print(f"True drifted features: {n_truly_drifted}")
print(f"Detected (raw, alpha=0.05): {detected_raw}")
print(f"Detected (Bonferroni): {sum(reject_bonf)}")
print(f"Detected (Benjamini-Hochberg): {sum(reject_bh)}")
```

### Exercise 3: Categorical Drift with Chi-Squared

**Goal:** Detect drift in categorical features.

```python
import pandas as pd

# Reference: category distribution
reference_cat = pd.Series(
    np.random.choice(["A", "B", "C", "D"], size=5000, p=[0.4, 0.3, 0.2, 0.1])
)

# Current: shifted distribution + new category
current_cat = pd.Series(
    np.random.choice(["A", "B", "C", "D", "E"], size=1000, p=[0.2, 0.2, 0.3, 0.2, 0.1])
)

result = chi_squared_drift_test(reference_cat, current_cat)
print(f"Chi-squared stat: {result['chi2_statistic']:.4f}")
print(f"P-value: {result['p_value']:.6f}")
print(f"Cramer's V: {result['cramers_v']:.4f}")
print(f"Drifted: {result['is_drifted']}")
print(f"New categories: {result['new_categories']}")
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No multiple testing correction | Too many false positives | Apply Bonferroni or BH correction |
| Using parametric tests on non-normal data | Invalid p-values | Use non-parametric tests (KS, Mann-Whitney) |
| Ignoring effect size | Everything is "significant" | Report effect size alongside p-value |
| Same alpha for all features | Important features treated equally to noise features | Weight alpha by feature importance |
| Not logging test configurations | Cannot reproduce results | Log alpha, test type, and sample sizes |

---

## Self-Check Questions

1. What is the null hypothesis in a KS test for drift detection?
2. Why does the KS test become unreliable with very large samples?
3. When should you use Chi-squared vs. PSI for categorical features?
4. What is the difference between Bonferroni and Benjamini-Hochberg correction?
5. How do you choose between a parametric and non-parametric test?

---

## You Know You Have Completed This Module When...

- [ ] You can apply KS, Chi-squared, and PSI tests to detect drift
- [ ] You understand p-values, significance levels, and effect sizes
- [ ] You can apply multiple testing correction to multi-feature datasets
- [ ] Validation script passes: `bash modules/04-statistical-tests/validation/validate.sh`
- [ ] You can justify your choice of statistical test in a code review

---

## Troubleshooting

### Common Issues

**Issue: `statsmodels` not installed**
```bash
pip install statsmodels
```

**Issue: Chi-squared test warns about expected frequencies**
- Small expected cell counts violate test assumptions
- Use Fisher's exact test for small samples

**Issue: p-value is exactly 0.0**
- Very large sample size causes the p-value to underflow
- Report as "p < 1e-300" and focus on effect size

---

**Next: [Module 05 -- Model Performance Tracking ->](../05-model-performance-tracking/)**
