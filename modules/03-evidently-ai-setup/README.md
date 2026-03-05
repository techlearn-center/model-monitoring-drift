# Module 03: Evidently AI -- Setup, Reports, Dashboards, and Test Suites

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 02 completed, Python 3.10+ |

---

## Learning Objectives

By the end of this module, you will be able to:

- Install and configure the Evidently AI framework for model monitoring
- Generate data drift, model performance, and data quality reports
- Create interactive HTML dashboards for stakeholder communication
- Build test suites with pass/fail criteria for CI/CD integration
- Use column mapping to handle different data types and roles

---

## Concepts

### What Is Evidently AI?

Evidently is an open-source Python library for evaluating, testing, and monitoring ML models in production. It provides pre-built metrics, reports, and test suites that cover the most common monitoring needs without requiring you to implement statistical tests from scratch.

**Why Evidently matters:** Instead of building drift detection, performance tracking, and report generation from scratch, Evidently gives you production-ready components that integrate with your existing ML pipeline.

### Evidently Architecture

```
                          +------------------+
                          |  Column Mapping  |
                          |  (data schema)   |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
    +---------v--------+  +-------v--------+  +---------v--------+
    |    Reports       |  |  Test Suites   |  |    Monitors      |
    |  (exploration)   |  |  (CI/CD)       |  |  (production)    |
    +------------------+  +----------------+  +------------------+
    |  - DataDriftPreset|  | - TestColumnDrift|  | - Prometheus    |
    |  - DataQuality   |  | - TestShareOf   |  | - Grafana       |
    |  - Classification|  |   DriftedColumns|  | - Scheduled     |
    |  - Regression    |  | - TestAccuracy  |  |   checks        |
    +------------------+  +----------------+  +------------------+
```

### Core Components

#### 1. Reports (Exploration and Analysis)

Reports generate detailed HTML or JSON output for a single analysis run. They are ideal for ad-hoc investigation, stakeholder communication, and deep-dive analysis.

**Available Presets:**
| Preset | Purpose | Key Metrics |
|---|---|---|
| `DataDriftPreset` | Detect input data drift | Per-column drift scores, dataset drift share |
| `DataQualityPreset` | Check data quality issues | Missing values, duplicates, out-of-range |
| `ClassificationPreset` | Classification model performance | Accuracy, F1, AUC, confusion matrix |
| `RegressionPreset` | Regression model performance | MSE, MAE, R2, residual analysis |
| `TargetDriftPreset` | Detect target/prediction drift | Target distribution changes |

#### 2. Test Suites (CI/CD and Automated Checks)

Test suites produce pass/fail results that integrate with CI/CD pipelines. Each test checks a specific condition and returns a boolean result.

```python
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestMeanInNSigmas,
    TestNumberOfMissingValues,
)

suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.3),        # Less than 30% drifted
    TestColumnDrift(column_name="age"),         # Check specific column
    TestMeanInNSigmas(column_name="income", n=3),  # Mean within 3 sigma
    TestNumberOfMissingValues(column_name="score", eq=0),
])
```

#### 3. Column Mapping

Column mapping tells Evidently how to interpret your data -- which columns are features, which is the target, which is the prediction, and their data types.

```python
from evidently import ColumnMapping

column_mapping = ColumnMapping()
column_mapping.target = "target"
column_mapping.prediction = "prediction"
column_mapping.numerical_features = ["age", "income", "score"]
column_mapping.categorical_features = ["gender", "region", "plan_type"]
column_mapping.datetime = "timestamp"
column_mapping.id = "user_id"
```

### Report Output Formats

| Format | Use Case | Method |
|---|---|---|
| HTML | Visual dashboards, stakeholder review | `report.save_html("report.html")` |
| JSON | Programmatic analysis, storage | `report.json()` |
| Dictionary | Python-native processing | `report.as_dict()` |
| Notebook | Interactive Jupyter display | `report.show()` |

---

## Hands-On Lab

### Prerequisites Check

```bash
pip install evidently>=0.4.0 pandas scikit-learn
python -c "import evidently; print(f'Evidently version: {evidently.__version__}')"
```

### Exercise 1: Generate a Data Drift Report

**Goal:** Create your first Evidently drift report.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load and split data
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

# Split into reference and current (simulate production)
reference = df.iloc[:100]
current = df.iloc[100:]

# Add synthetic drift to current data
current_drifted = current.copy()
current_drifted["sepal_length"] = current_drifted["sepal_length"] + np.random.normal(1.5, 0.3, len(current_drifted))

# Configure column mapping
column_mapping = ColumnMapping()
column_mapping.numerical_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
column_mapping.target = "target"

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=reference,
    current_data=current_drifted,
    column_mapping=column_mapping,
)

# Save as HTML
report.save_html("reports/data_drift_report.html")
print("Report saved! Open reports/data_drift_report.html in your browser.")

# Access results programmatically
result = report.as_dict()
for metric in result["metrics"]:
    metric_id = metric["metric"]
    print(f"\nMetric: {metric_id}")
    if "result" in metric:
        for key, val in metric["result"].items():
            if isinstance(val, (int, float, bool)):
                print(f"  {key}: {val}")
```

### Exercise 2: Build a Test Suite for CI/CD

**Goal:** Create automated pass/fail checks suitable for CI/CD pipeline gates.

```python
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
    TestMeanInNSigmas,
    TestNumberOfMissingValues,
    TestShareOfMissingValues,
)

# Build test suite
suite = TestSuite(tests=[
    # Dataset-level: no more than 30% of columns should drift
    TestShareOfDriftedColumns(lt=0.3),

    # Column-level drift checks
    TestColumnDrift(column_name="sepal_length"),
    TestColumnDrift(column_name="sepal_width"),
    TestColumnDrift(column_name="petal_length"),
    TestColumnDrift(column_name="petal_width"),

    # Data quality checks
    TestNumberOfMissingValues(column_name="sepal_length", eq=0),
    TestMeanInNSigmas(column_name="sepal_width", n=3),
])

# Run against production data
suite.run(
    reference_data=reference,
    current_data=current_drifted,
    column_mapping=column_mapping,
)

# Check results
suite_result = suite.as_dict()
print(f"\nTest Suite Summary:")
print(f"  Total tests: {suite_result['summary']['total_tests']}")
print(f"  Passed: {suite_result['summary']['success_tests']}")
print(f"  Failed: {suite_result['summary']['failed_tests']}")

# In CI/CD, use the return code
all_passed = suite_result["summary"]["all_passed"]
print(f"  All passed: {all_passed}")

if not all_passed:
    print("\nFailed tests:")
    for test in suite_result["tests"]:
        if test["status"] == "FAIL":
            print(f"  - {test['name']}: {test['description']}")
```

### Exercise 3: Multi-Preset Dashboard

**Goal:** Combine multiple report presets into a comprehensive monitoring dashboard.

```python
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)

# Comprehensive report with multiple presets
comprehensive_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset(),
])

comprehensive_report.run(
    reference_data=reference,
    current_data=current_drifted,
    column_mapping=column_mapping,
)

comprehensive_report.save_html("reports/comprehensive_report.html")
print("Comprehensive report saved!")
```

---

## Starter Files

Check `lab/starter/` for:
- `setup_evidently.py` -- Evidently installation verification
- `report_template.py` -- Report generation skeleton
- `test_suite_template.py` -- Test suite skeleton for CI/CD
- `sample_data/` -- Pre-built datasets with known drift

## Solution Files

If you get stuck, `lab/solution/` contains:
- Complete report generation scripts
- CI/CD-ready test suite configurations
- Sample HTML reports for reference

> **Important:** Try to complete the exercises yourself first!

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting column mapping | Evidently guesses types incorrectly | Always specify numerical_features and categorical_features |
| Not setting target/prediction columns | Classification/regression presets fail | Set column_mapping.target and column_mapping.prediction |
| Reference and current have different columns | KeyError or silent failures | Ensure both DataFrames share the same schema |
| Using too small a reference dataset | Unstable drift detection | Use at least 1000 samples as reference |
| Not saving JSON alongside HTML | Cannot process results programmatically | Always call `.as_dict()` or `.json()` alongside HTML |

---

## Self-Check Questions

1. What is the difference between an Evidently Report and a Test Suite?
2. When would you use `DataDriftPreset` vs. `TargetDriftPreset`?
3. How do you integrate Evidently test suites into a CI/CD pipeline?
4. What does the `column_mapping` parameter control, and why is it important?
5. How would you generate Evidently reports on a recurring schedule in production?

---

## You Know You Have Completed This Module When...

- [ ] You can generate HTML drift reports using Evidently
- [ ] You have built a test suite with pass/fail criteria
- [ ] You understand how to configure column mapping for your data
- [ ] You can extract drift results programmatically from report JSON
- [ ] Validation script passes: `bash modules/03-evidently-ai-setup/validation/validate.sh`
- [ ] You can explain Evidently's role in a production monitoring stack

---

## Troubleshooting

### Common Issues

**Issue: `ImportError: cannot import name 'DataDriftPreset'`**
```bash
pip install --upgrade evidently
# Evidently 0.4+ uses evidently.metric_preset
```

**Issue: Report HTML is blank or incomplete**
- Check that reference and current data have matching columns
- Verify column mapping is correct
- Try `report.as_dict()` to check for error messages

**Issue: Test suite shows unexpected failures**
- Review threshold values -- they may be too strict
- Check sample sizes -- small samples produce unreliable tests

---

**Next: [Module 04 -- Statistical Tests for Drift ->](../04-statistical-tests/)**
