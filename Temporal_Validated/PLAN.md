# PLAN: Temporal Validation Rebuild

## Executive Summary

This folder (`Temporal_Validated/`) contains a complete rebuild of the NCAA Financial Trajectory Prediction project. The original `final/` folder achieved 87-92% accuracy, but this result was **artificially inflated due to data leakage**. This rebuild implements proper temporal validation to produce honest, trustworthy predictions.

**Key Insight:** A 55-65% accuracy that is *real* is far more valuable than a 92% that is *fake*.

---

## Part 1: Problems Identified in `final/`

### Problem 1: Lagged Target Labels Used as Features (CRITICAL)

**Location:** `final/assets/notebooks/03_Feature_Engineering_Advanced.ipynb`

**The Problematic Code:**
```python
# Lag columns - only use columns that exist in the dataset
lag_columns = [
    'Target_Label',  # <-- THIS IS THE PROBLEM
    'Efficiency_Mean_2yr', 'Revenue_Growth_1yr', ...
]

for col in lag_columns:
    if col in df.columns:
        df[f'Lag1_{col}'] = grouped[col].shift(1)

# Additional leakage features created:
df['Same_Trajectory_As_Lag'] = same_traj.astype(int)
df['Trajectory_Changed'] = 1 - df['Same_Trajectory_As_Lag']
df['Lag1_Target_Declining'] = (df['Lag1_Target_Label'] == 0).fillna(False).astype(int)
df['Lag1_Target_Stable'] = (df['Lag1_Target_Label'] == 1).fillna(False).astype(int)
df['Lag1_Target_Improving'] = (df['Lag1_Target_Label'] == 2).fillna(False).astype(int)
```

**Why This Causes Leakage:**
- `Target_Label[t]` is calculated from the transition from year t to year t+1
- `Lag1_Target_Label[t]` = `Target_Label[t-1]` = transition from year t-1 to year t
- So for a row at year t, `Lag1_Target_Label` contains information about what happened FROM t-1 TO t
- This overlaps with the current row's timeframe, leaking "future" information into features

**Example:**
| Row Year | Lag1_Target (Feature) | Target (What We Predict) |
|----------|----------------------|--------------------------|
| 2020 | Transition 2019→2020 | Transition 2020→2021 |
| 2021 | Transition 2020→2021 | Transition 2021→2022 |

The feature contains information about growth rates that overlap with what we're predicting.

---

### Problem 2: Random Train/Test Split Instead of Temporal

**Location:** `final/assets/notebooks/04_Model_Training.ipynb`

**The Problematic Code:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # RANDOM split
)
```

**Why This Is Wrong:**
- A 2021 row might be in TEST while a 2022 row from the SAME institution is in TRAIN
- The model learns from 2022 to "predict" 2021 — predicting the past
- Information about future years leaks into training through institutional patterns
- This is NOT how the model would be used in production (we predict forward, not backward)

---

### Problem 3: 2023 Data Dropped Instead of Used as Holdout

**Location:** `final/assets/notebooks/01_Feature_Engineering_Basic.ipynb`

**What Happened:**
- Raw data contains 2014-2023 (10 years)
- Final dataset only contains 2017-2022 (6 years)
- 2023 rows were dropped because they have `NaN` targets (no 2024 data to calculate future growth)

**Why This Is a Problem:**
- 2023 is the PERFECT true holdout — it has features but no target
- We should use 2022 features to predict 2023 outcomes
- Instead, the model was never tested on truly unseen future data

---

## Part 2: Our Solution

### Approach: Fresh Start with Organic EDA

We treat this as a completely new project:
1. Load raw data as if seeing it for the first time
2. Explore through genuine EDA
3. Discover features organically based on observations
4. Build only leak-free features
5. Apply strict temporal validation
6. Report honest metrics

---

### Year Allocation

```
Raw Data: 2014-2023 (10 years, 17,220 records)

Timeline:
2014    2015    2016    2017    2018    2019    2020    2021    2022    2023
|_______|_______|       |_______|_______|_______|_______|_______|_______|
        ↓                       ↓               ↓       ↓       ↓
    LOOKBACK              TRAINING          VALIDATION  HOLDOUT  FUTURE
    (consumed for         (learn            (tune       (final   (true
    2-year lags)          patterns)         params)     test)    predictions)
```

| Years | Role | Rows in Dataset? |
|-------|------|------------------|
| 2014-2016 | Consumed for 2-year lag feature calculations | ❌ No (used as lookback only) |
| 2017-2019 | Training set | ✅ Yes (~5,166 rows) |
| 2020-2021 | Validation set (hyperparameter tuning) | ✅ Yes (~3,444 rows) |
| 2022 | Holdout test set (final honest evaluation) | ✅ Yes (~1,722 rows) |
| 2023 | True future predictions (no labels available) | ✅ Yes (~1,722 rows, no target) |

---

### Feature Rules

#### ✅ ALLOWED Features (Use Only Past Data)

| Feature | Formula | Why It's Safe |
|---------|---------|---------------|
| `Revenue_Growth_1yr` | `(Rev[t] - Rev[t-1]) / Rev[t-1]` | Uses t and t-1 only |
| `Expense_Growth_1yr` | `(Exp[t] - Exp[t-1]) / Exp[t-1]` | Uses t and t-1 only |
| `Revenue_CAGR_2yr` | `(Rev[t] / Rev[t-2])^0.5 - 1` | Uses t, t-1, t-2 only |
| `Expense_CAGR_2yr` | `(Exp[t] / Exp[t-2])^0.5 - 1` | Uses t, t-1, t-2 only |
| `Efficiency_Ratio` | `Rev[t] / Exp[t]` | Current year only |
| `Efficiency_Mean_2yr` | `mean(Efficiency[t], Efficiency[t-1])` | Past data only |
| `Revenue_Volatility_2yr` | `std(Rev[t], Rev[t-1])` | Past data only |
| `Revenue_Per_Athlete` | `Rev[t] / Athletes[t]` | Current year only |
| `Division`, `State` | Categorical | Structural, no time component |

#### ❌ FORBIDDEN Features (Cause Leakage)

| Feature | Why Forbidden |
|---------|---------------|
| `Lag1_Target_Label` | Derived from t-1→t transition (overlaps with prediction timeframe) |
| `Same_Trajectory_As_Lag` | Compares current target to leaked lag feature |
| `Trajectory_Changed` | Same issue as above |
| `Lag1_Target_Declining` | One-hot encoding of leaked label |
| `Lag1_Target_Stable` | One-hot encoding of leaked label |
| `Lag1_Target_Improving` | One-hot encoding of leaked label |
| ANY feature derived from `Target_Label` | By definition, contains future information |

---

### Target Definition

The target captures financial trajectory over the NEXT year (t → t+1):

```python
# Calculate future growth (what happens AFTER year t)
Future_Revenue_Growth = (Revenue[t+1] - Revenue[t]) / Revenue[t]
Future_Expense_Growth = (Expense[t+1] - Expense[t]) / Expense[t]

# Define trajectory classes
IMPROVING:  Future_Revenue_Growth > 3% AND Future_Expense_Growth < Future_Revenue_Growth
DECLINING:  Future_Revenue_Growth < 0% OR Future_Expense_Growth > Future_Revenue_Growth + 3%
STABLE:     Everything else
```

This is correct — the target SHOULD look at the future. The problem was using lagged TARGETS as features.

---

## Part 3: Verification Checkpoints

Each notebook includes verification cells to catch any remaining issues:

| Checkpoint | Notebook | What It Verifies |
|------------|----------|------------------|
| Feature Audit | 04_Feature_Engineering | No column contains "Target" except actual target |
| Year Leak Check | 05_Temporal_Split | Train years < Validation years < Holdout year |
| No Future Data | 04_Feature_Engineering | All features use `shift(1)` or `shift(2)`, never `shift(-1)` |
| Accuracy Sanity | 07_Model_Evaluation | If accuracy > 80%, raise warning for investigation |
| Baseline Comparison | 07_Model_Evaluation | Model beats random (33%) and majority class (~46%) |

---

## Part 4: Expected Outcomes

| Metric | `final/` (Inflated) | `Temporal_Validated/` (Honest) |
|--------|---------------------|--------------------------------|
| Accuracy | 87-92% | 55-65% |
| ROC-AUC | 0.96-0.97 | 0.65-0.75 |
| Macro F1 | 0.82-0.83 | 0.45-0.55 |
| Improving F1 | 0.65-0.66 | 0.30-0.45 |

**Why lower is better:**
- The honest metrics represent actual predictive power
- 55-65% accuracy is still ~20-30% above random baseline (33%)
- This demonstrates the model learned real patterns, not just a formula

---

## Part 5: Notebook Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  00_Data_Loading_and_Overview.ipynb                                     │
│  ─────────────────────────────────────────                              │
│  Input:  data/raw/Output_10yrs_reported_schools_17220.csv               │
│  Output: Understanding of data structure, columns, years                │
│  Next:   "Let's explore this data in depth"                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  01_Exploratory_Data_Analysis.ipynb                                     │
│  ─────────────────────────────────────────                              │
│  Input:  Raw data                                                       │
│  Output: Observations about distributions, correlations, patterns       │
│  Next:   "Based on these observations, here are feature ideas"          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  02_Feature_Discovery.ipynb                                             │
│  ─────────────────────────────────────────                              │
│  Input:  EDA observations                                               │
│  Output: List of proposed features with justification                   │
│  Next:   "Before building features, let's define our target"            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  03_Target_Definition.ipynb                                             │
│  ─────────────────────────────────────────                              │
│  Input:  Raw data                                                       │
│  Output: Target variable definition, threshold justification            │
│  Next:   "Now let's build the features"                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  04_Feature_Engineering.ipynb                                           │
│  ─────────────────────────────────────────                              │
│  Input:  Raw data + feature list + target definition                    │
│  Output: data/processed/features_temporal.csv                           │
│  Next:   "Let's split this data properly by time"                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  05_Temporal_Split.ipynb                                                │
│  ─────────────────────────────────────────                              │
│  Input:  data/processed/features_temporal.csv                           │
│  Output: Train/Validation/Holdout/Future split datasets                 │
│  Next:   "Let's train models on the training set"                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  06_Model_Training.ipynb                                                │
│  ─────────────────────────────────────────                              │
│  Input:  Train and Validation splits                                    │
│  Output: Trained models in models/ folder                               │
│  Next:   "Let's evaluate on the holdout set"                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  07_Model_Evaluation.ipynb                                              │
│  ─────────────────────────────────────────                              │
│  Input:  Trained models + Holdout set (2022)                            │
│  Output: Honest metrics, comparison to baselines                        │
│  Next:   "Let's predict the true future (2023)"                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  08_Predictions.ipynb                                                   │
│  ─────────────────────────────────────────                              │
│  Input:  Trained models + 2023 data (no labels)                         │
│  Output: Predictions for 2023, institution-level reports                │
│  Final:  "These are true predictions for unseen future data"            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Summary

| Aspect | `final/` (Before) | `Temporal_Validated/` (After) |
|--------|-------------------|-------------------------------|
| Approach | Fix existing code | Fresh start with organic EDA |
| Features | Used `Lag1_Target_Label` | Explicitly forbidden |
| Split | Random 80/20 | Temporal: 2017-2019 / 2020-2021 / 2022 / 2023 |
| 2023 Data | Dropped | Preserved as true holdout |
| Documentation | Minimal | Every cell explains WHAT and WHY |
| Accuracy | 87-92% (fake) | 55-65% (honest) |
| Trust Level | ❌ Misleading | ✅ Reliable |

---

## Conclusion

This rebuild addresses the fundamental flaws in the original project:

1. **No data leakage** — Features use only past data
2. **Proper temporal split** — Train on past, test on future
3. **True future prediction** — 2023 data used as real holdout
4. **Full documentation** — Every decision explained
5. **Verification built-in** — Checkpoints catch any remaining issues

The resulting model will have lower accuracy, but that accuracy will be **real** and **actionable**.
