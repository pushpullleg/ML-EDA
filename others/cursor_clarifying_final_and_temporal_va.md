# Clarifying final and temporal validation differences
_Exported on 11/28/2025 at 22:02:52 CST from Cursor (2.1.39)_

---

**User**

Okay, I was working on the final folder, and that is what we were deciding was the final submission. However, later I did ask something, and it said there is a circular kind of validation - data leakage is what I found. What I did was, I entirely built a new project, Temporal Validation. Okay. And I am just giving you the context. What I found in chatting with other agents was the difference between Final and Temporal Validation folders. There is a problem in the training split, for example, this temporal that is a random training split is what I understood. So can you go ahead and check that? Confirm what's the primary difference between these two? And also prove or tell me where the differences?

---

**Cursor**

I'll help you understand the differences between the `final/` and `Temporal_Validated/` folders, particularly around training splits and data leakage. Let me start by exploring both directories.

Now let me examine the key notebooks that handle the training/validation splits in both projects. I'll look at the model training notebooks and specifically the temporal split notebook.

I found significant information. Now let me also check the feature engineering in both projects to understand the complete picture of where the data leakage originated.

Now I have a complete picture. Let me summarize the **critical differences** between these two approaches:

---

## 🔍 Primary Difference: Two Major Sources of Data Leakage in `final/`

### **1. Random Split vs Temporal Split (Training Strategy)**

| Aspect | `final/` | `Temporal_Validated/` |
|--------|----------|----------------------|
| **Split Method** | `train_test_split(..., stratify=y, random_state=42)` | Split by Year boundaries |
| **Training Data** | Random 80% from 2017-2022 (mixed years) | Only 2017, 2018, 2019 |
| **Test Data** | Random 20% from any year | Only 2022 |
| **Problem** | A 2020 row can train while its 2019 lag is in test | Training always uses PAST data only |

Here's the exact problematic code in `final/`:

```5:5:final/assets/notebooks/04_Model_Training.ipynb
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Versus the correct temporal approach in `Temporal_Validated/`:

```7:7:Temporal_Validated/notebooks/05_Temporal_Split.ipynb
# Training: 2017-2019
train_years = [2017, 2018, 2019]
df_train = df[df['Year'].isin(train_years)].copy()

# Validation: 2020-2021
val_years = [2020, 2021]
df_val = df[df['Year'].isin(val_years)].copy()

# Test: 2022
test_years = [2022]
df_test = df[df['Year'].isin(test_years)].copy()
```

---

### **2. Leaky Features (Target-Derived Features)**

The `final/` folder's `03_Feature_Engineering_Advanced.ipynb` creates **forbidden features** that leak the target:

```5:5:final/assets/notebooks/03_Feature_Engineering_Advanced.ipynb
# Persistence features - THESE LEAK THE TARGET!
df['Lag1_Target_Label'] = df['Lag1_Target_Label'].astype(pd.Int64Dtype())
same_traj = (df['Target_Label'] == df['Lag1_Target_Label']).fillna(False)
df['Same_Trajectory_As_Lag'] = same_traj.astype(int)
df['Trajectory_Changed'] = 1 - df['Same_Trajectory_As_Lag']

df['Lag1_Target_Declining'] = (df['Lag1_Target_Label'] == 0).fillna(False).astype(int)
df['Lag1_Target_Stable'] = (df['Lag1_Target_Label'] == 1).fillna(False).astype(int)
df['Lag1_Target_Improving'] = (df['Lag1_Target_Label'] == 2).fillna(False).astype(int)
```

**Why is this leakage?**
- If you know an institution's previous trajectory (`Lag1_Target_Label`), you have direct information about the target!
- Combined with random splitting, a 2019 row might be in training (with its 2018 target as a feature), while its 2018 row is in test (whose target IS that feature value)

In contrast, `Temporal_Validated/` explicitly forbids these:

```2:2:Temporal_Validated/notebooks/leakage_check.ipynb
# Define forbidden patterns
forbidden_patterns = [
    'Lag1_Target',
    'Lag2_Target',
    'Same_Trajectory',
    'Trajectory_Changed',
    'Future_',
    'Next_'
]
```

---

## 📊 The Proof: Accuracy Difference

| Metric | `final/` (Leakage) | `Temporal_Validated/` (Clean) |
|--------|-------------------|------------------------------|
| **Accuracy** | 87-88% | 53-57% |
| **ROC-AUC** | 0.96-0.97 | ~0.70 |
| **Interpretation** | Suspiciously high | Realistic for this problem |

From the leakage check notebook:

```8:8:Temporal_Validated/notebooks/leakage_check.ipynb
if 0.45 <= best_acc <= 0.75:
    print("\n✅ PASS: Accuracy is in realistic range (45-75%)")
    print("   This is a genuinely hard prediction problem.")
elif best_acc > 0.80:
    print("\n⚠️ WARNING: Accuracy is suspiciously high (>80%)")
    print("   This might indicate remaining data leakage.")
    print("   Compare to final/ which had 87-92% with leakage.")
```

---

## 🧠 Visual Summary of the Problem

```
FINAL/ (Random Split + Leaky Features):
=========================================
Data: 2017-2022 all mixed

Training Set (random 80%):
  ├── 2017 rows with Lag1_Target from 2016
  ├── 2018 rows with Lag1_Target from 2017  ← Uses 2017's target!
  ├── 2019 rows with Lag1_Target from 2018
  ├── 2020 rows with Lag1_Target from 2019
  └── 2021 rows with Lag1_Target from 2020

Test Set (random 20%):
  ├── 2017 rows  ← Its target is used as feature in 2018 training row!
  ├── 2018 rows  ← Its target is used as feature in 2019 training row!
  └── ... CIRCULAR DEPENDENCY!

TEMPORAL_VALIDATED/ (Temporal Split + Clean Features):
======================================================
Training: 2017-2019 (only PAST)
Validation: 2020-2021 (future of training)
Test: 2022 (future of everything)

No Lag1_Target features → No circular dependency
Model must predict WITHOUT knowing previous trajectory
```

---

## ✅ Conclusion

**You correctly identified the problem!** The `final/` folder has two types of data leakage:

1. **Random train/test split** allows future data to inform past predictions
2. **Target-derived features** (`Lag1_Target_Label`, `Same_Trajectory_As_Lag`) directly leak information about what the model should predict

The `Temporal_Validated/` folder fixes both issues:
1. Strict temporal split (train on 2017-2019, test on 2022)
2. Only uses financial lag features (revenue, expenses), never target lags

The ~57% accuracy in `Temporal_Validated/` is the **honest, realistic performance** for this prediction task.

```plaintext
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

```plaintext
# Training: 2017-2019
train_years = [2017, 2018, 2019]
df_train = df[df['Year'].isin(train_years)].copy()

# Validation: 2020-2021
val_years = [2020, 2021]
df_val = df[df['Year'].isin(val_years)].copy()

# Test: 2022
test_years = [2022]
df_test = df[df['Year'].isin(test_years)].copy()
```

```plaintext
# Persistence features - THESE LEAK THE TARGET!
df['Lag1_Target_Label'] = df['Lag1_Target_Label'].astype(pd.Int64Dtype())
same_traj = (df['Target_Label'] == df['Lag1_Target_Label']).fillna(False)
df['Same_Trajectory_As_Lag'] = same_traj.astype(int)
df['Trajectory_Changed'] = 1 - df['Same_Trajectory_As_Lag']

df['Lag1_Target_Declining'] = (df['Lag1_Target_Label'] == 0).fillna(False).astype(int)
df['Lag1_Target_Stable'] = (df['Lag1_Target_Label'] == 1).fillna(False).astype(int)
df['Lag1_Target_Improving'] = (df['Lag1_Target_Label'] == 2).fillna(False).astype(int)
```

```plaintext
# Define forbidden patterns
forbidden_patterns = [
    'Lag1_Target',
    'Lag2_Target',
    'Same_Trajectory',
    'Trajectory_Changed',
    'Future_',
    'Next_'
]
```

```plaintext
if 0.45 <= best_acc <= 0.75:
    print("\n✅ PASS: Accuracy is in realistic range (45-75%)")
    print("   This is a genuinely hard prediction problem.")
elif best_acc > 0.80:
    print("\n⚠️ WARNING: Accuracy is suspiciously high (>80%)")
    print("   This might indicate remaining data leakage.")
    print("   Compare to final/ which had 87-92% with leakage.")
```

```plaintext
FINAL/ (Random Split + Leaky Features):
=========================================
Data: 2017-2022 all mixed

Training Set (random 80%):
  ├── 2017 rows with Lag1_Target from 2016
  ├── 2018 rows with Lag1_Target from 2017  ← Uses 2017's target!
  ├── 2019 rows with Lag1_Target from 2018
  ├── 2020 rows with Lag1_Target from 2019
  └── 2021 rows with Lag1_Target from 2020

Test Set (random 20%):
  ├── 2017 rows  ← Its target is used as feature in 2018 training row!
  ├── 2018 rows  ← Its target is used as feature in 2019 training row!
  └── ... CIRCULAR DEPENDENCY!

TEMPORAL_VALIDATED/ (Temporal Split + Clean Features):
======================================================
Training: 2017-2019 (only PAST)
Validation: 2020-2021 (future of training)
Test: 2022 (future of everything)

No Lag1_Target features → No circular dependency
Model must predict WITHOUT knowing previous trajectory
```

---

**User**

In the temporal where is it validated? Where is year 2021? Is it on the training or it is on testing or it is left alone? What is done?

---

**Cursor**

Great question! Let me show you exactly where 2021 is used in the `Temporal_Validated/` approach.

## 2021 is in the **Validation Set**

From the temporal split notebook:

```5:5:Temporal_Validated/notebooks/05_Temporal_Split.ipynb
╔══════════════════════════════════════════════════════════════════════════════╗
║                            TEMPORAL SPLIT PLAN                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TRAINING SET: 2017-2019                                                     ║
║  ─────────────────────────                                                   ║
║  • Purpose: Learn patterns                                                   ║
║  • Years: 2017, 2018, 2019                                                   ║
║  • Expected: ~5,166 rows (1,722 × 3)                                         ║
║                                                                              ║
║  VALIDATION SET: 2020-2021                                                   ║
║  ────────────────────────────                                                ║
║  • Purpose: Tune hyperparameters                                             ║
║  • Years: 2020, 2021                                                         ║
║  • Expected: ~3,444 rows (1,722 × 2)                                         ║
║  • Note: Includes COVID years (may affect patterns)                          ║
║                                                                              ║
║  TEST SET: 2022                                                              ║
║  ─────────────────────                                                       ║
║  • Purpose: Final unbiased evaluation                                        ║
║  • Years: 2022                                                               ║
║  • Expected: ~1,722 rows                                                     ║
║  • NEVER used for training or tuning                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Visual Timeline

```
Year:     2017    2018    2019    2020    2021    2022    2023
          ├───────────────────┤├─────────────┤├─────┤├─────┤
Purpose:       TRAINING            VALIDATION    TEST   PREDICT
          (learn patterns)    (tune hyperparams) (final) (future)
Rows:         5,166 rows          3,444 rows    1,722   1,722
```

## What Each Set Does

| Set | Years | Purpose | Used For |
|-----|-------|---------|----------|
| **Training** | 2017, 2018, 2019 | Learn patterns | Model fitting (`.fit()`) |
| **Validation** | **2020, 2021** | Tune hyperparameters | Early stopping, model selection |
| **Test** | 2022 | Final evaluation | Report final metrics (untouched during tuning) |
| **Prediction** | 2023 | True future | No labels exist; real predictions |

## Why 2021 is in Validation (not Training or Test)

1. **Not in Training**: Because we want to validate on "future" data that the model hasn't seen
2. **Not in Test**: The test set (2022) is kept completely separate for final, unbiased evaluation
3. **In Validation**: 2020-2021 serves as an intermediate holdout to:
   - Tune hyperparameters
   - Implement early stopping
   - Compare different models
   - Without "contaminating" the final test set

The class distribution output confirms this:

```15:15:Temporal_Validated/notebooks/05_Temporal_Split.ipynb
TRAINING (2017-2019):
   Declining: 2,787 (53.9%)
   Stable: 1,844 (35.7%)
   Improving: 535 (10.4%)

VALIDATION (2020-2021):    ← 2021 is here
   Declining: 871 (25.3%)
   Stable: 1,975 (57.3%)
   Improving: 598 (17.4%)

TEST (2022):
   Declining: 487 (28.3%)
   Stable: 972 (56.4%)
   Improving: 263 (15.3%)
```

So **2021 is used for validation/hyperparameter tuning** — it's neither in training nor in the final test set.

```plaintext
╔══════════════════════════════════════════════════════════════════════════════╗
║                            TEMPORAL SPLIT PLAN                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TRAINING SET: 2017-2019                                                     ║
║  ─────────────────────────                                                   ║
║  • Purpose: Learn patterns                                                   ║
║  • Years: 2017, 2018, 2019                                                   ║
║  • Expected: ~5,166 rows (1,722 × 3)                                         ║
║                                                                              ║
║  VALIDATION SET: 2020-2021                                                   ║
║  ────────────────────────────                                                ║
║  • Purpose: Tune hyperparameters                                             ║
║  • Years: 2020, 2021                                                         ║
║  • Expected: ~3,444 rows (1,722 × 2)                                         ║
║  • Note: Includes COVID years (may affect patterns)                          ║
║                                                                              ║
║  TEST SET: 2022                                                              ║
║  ─────────────────────                                                       ║
║  • Purpose: Final unbiased evaluation                                        ║
║  • Years: 2022                                                               ║
║  • Expected: ~1,722 rows                                                     ║
║  • NEVER used for training or tuning                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

```plaintext
Year:     2017    2018    2019    2020    2021    2022    2023
          ├───────────────────┤├─────────────┤├─────┤├─────┤
Purpose:       TRAINING            VALIDATION    TEST   PREDICT
          (learn patterns)    (tune hyperparams) (final) (future)
Rows:         5,166 rows          3,444 rows    1,722   1,722
```

```plaintext
TRAINING (2017-2019):
   Declining: 2,787 (53.9%)
   Stable: 1,844 (35.7%)
   Improving: 535 (10.4%)

VALIDATION (2020-2021):    ← 2021 is here
   Declining: 871 (25.3%)
   Stable: 1,975 (57.3%)
   Improving: 598 (17.4%)

TEST (2022):
   Declining: 487 (28.3%)
   Stable: 972 (56.4%)
   Improving: 263 (15.3%)
```

