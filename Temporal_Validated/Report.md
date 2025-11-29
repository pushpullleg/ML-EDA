# NCAA Athletics Financial Trajectory Prediction Report

## Executive Summary

This report documents a complete rebuild of the NCAA athletics financial trajectory prediction project. The original `final/` project achieved 87-92% accuracy but contained critical data leakage that invalidated all results. This project (`Temporal_Validated`) implements proper temporal validation and achieves honest accuracy in the 55-65% range — a genuine reflection of this prediction problem's difficulty.

---

## 1. The Problem with `final/`

### 1.1 What Went Wrong

The original project had **circular validation** — the model was essentially given the answers during training.

#### Data Leakage Source #1: Target-Derived Features
```python
# From final/ Feature Engineering
df['Lag1_Target_Label'] = df.groupby('UNITID')['Target_Label'].shift(1)
```

**Why this is wrong**: `Lag1_Target_Label` is the previous year's answer. Using it as a feature is like predicting tomorrow's weather by looking at tomorrow's weather report shifted by one day — it directly encodes the pattern you're trying to predict.

#### Data Leakage Source #2: Random Split
```python
# From final/ Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why this is wrong**: Random split allows 2022 data into training and 2017 data into testing. The model learns future patterns and "predicts" the past — achieving artificially high accuracy.

### 1.2 The Result

| Metric | Reported | Reality |
|--------|----------|---------|
| Accuracy | 87-92% | Inflated by leakage |
| 2023 Predictions | Dropped | Never generated |
| Validity | Claimed | None |

---

## 2. The Solution: Temporal Validation

### 2.1 Temporal Split Design

We split data **strictly by time** so the model can never "see the future":

```
2014 ─────────────────────────────────────────────────────→ 2023
  │                                                           │
  └──[LAGS: 2014-2016]──┬──[TRAIN: 2017-2019]──┬──[VAL: 2020-2021]──┬──[TEST: 2022]──┬──[PREDICT: 2023]
                        │                       │                     │                │
                    Learn patterns       Tune hyperparams        Final eval      True predictions
```

### 2.2 Feature Engineering Rules

#### Allowed Features (Use Past Data Only)
| Feature | Formula | Why Safe |
|---------|---------|----------|
| Revenue_Growth_1yr | (Rev[t] - Rev[t-1]) / Rev[t-1] | Uses only current and past |
| Revenue_CAGR_2yr | (Rev[t] / Rev[t-2])^0.5 - 1 | Uses only current and past |
| Efficiency_Ratio | Rev[t] / Exp[t] | Current year only |

#### Forbidden Features (Cause Leakage)
| Feature | Why Forbidden |
|---------|---------------|
| Lag1_Target_Label | Directly encodes answer |
| Same_Trajectory_As_Lag | Derived from target |
| Future_Revenue_Growth | Looks at future |

---

## 3. Methodology

### 3.1 Data

- **Source**: NCAA EADA Financial Data
- **Years**: 2014-2023 (10 years)
- **Institutions**: ~1,722 NCAA schools
- **Records**: 17,220 total

### 3.2 Target Variable

We predict financial trajectory from year t to year t+1:

| Class | Definition |
|-------|------------|
| **Improving** | Revenue growth > 3% AND Expense growth < Revenue growth |
| **Declining** | Revenue growth < 0% OR Expense growth > Revenue growth + 3% |
| **Stable** | Everything else |

### 3.3 Features (14 Total)

| Category | Features |
|----------|----------|
| Structural | Division |
| Raw Values | Grand Total Revenue, Grand Total Expenses |
| Current Metrics | Total_Athletes, Efficiency_Ratio, Revenue_Per_Athlete, Reports_Exactly_One |
| 1-Year Growth | Revenue_Growth_1yr, Expense_Growth_1yr |
| 2-Year Trends | Revenue_CAGR_2yr, Expense_CAGR_2yr, Efficiency_Mean_2yr |
| Volatility | Revenue_Volatility_2yr, Expense_Volatility_2yr |

### 3.4 Models

| Model | Hyperparameters |
|-------|-----------------|
| Logistic Regression | balanced class weights, multinomial, L-BFGS |
| Random Forest | 100 trees, max_depth=10, balanced weights |
| XGBoost | 100 estimators, max_depth=5, learning_rate=0.1 |

### 3.5 Class Imbalance

Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance training classes.

---

## 4. Results

### 4.1 Model Performance on 2022 Holdout

| Model | Accuracy | F1 (weighted) | F1 (macro) |
|-------|----------|---------------|------------|
| **Logistic Regression** | **57.3%** | **0.563** | **0.487** |
| Random Forest | 54.6% | 0.560 | 0.498 |
| XGBoost | 53.7% | 0.551 | 0.491 |

### 4.2 Comparison to Baseline

| Approach | Accuracy |
|----------|----------|
| Random Guessing | 33.3% |
| Most-Frequent Class (Declining) | 28.3% |
| **Our Best Model (Logistic Regression)** | **57.3%** |
| Improvement over baseline | **+29.0%** |
| final/ (with leakage) | 87-92% (invalid) |

### 4.3 Key Insight

**The lower accuracy is the correct answer.**

Financial trajectory prediction is inherently uncertain. A model that achieves 87-92% accuracy is almost certainly cheating. Our 57.3% accuracy represents:
- **+24%** improvement over random guessing (33.3%)
- **+29%** improvement over most-frequent-class baseline
- **Genuine predictive value** for stakeholder decisions
- **Honest uncertainty** that reflects reality

---

## 5. 2023 Predictions

### 5.1 Overview

| Predicted Trajectory | Count | Percentage |
|---------------------|-------|------------|
| Stable | 803 | 46.6% |
| Declining | 473 | 27.5% |
| Improving | 446 | 25.9% |

**Total institutions predicted: 1,722**

### 5.2 High-Confidence Predictions

Institutions with confidence score ≥ 70% are most actionable:
- **45 institutions** have predictions with ≥70% confidence
- See `reports/predictions_2023_high_confidence.csv` for details

#### Top High-Confidence Declining Predictions:
| Institution | Confidence |
|-------------|------------|
| Thomas University | 90.0% |
| University of Olivet | 89.0% |
| Miles College | 75.0% |

### 5.3 Important Note

These are **true predictions** — the answers don't exist yet. Validation would require 2024 financial data.

---

## 6. Verification

### 6.1 Leakage Checks Performed

| Check | Status | Details |
|-------|--------|----------|
| No target-derived features | ✅ Passed | 20 columns checked against 6 forbidden patterns |
| No future-looking features | ✅ Passed | No Future_ or Next_ prefixes found |
| Temporal split integrity | ✅ Passed | Train (2017-2019) < Val (2020-2021) < Test (2022) |
| No year overlap between splits | ✅ Passed | All 5 overlap checks passed |
| Year not used as feature | ✅ Passed | 14 clean feature columns |
| Accuracy in realistic range | ✅ Passed | 57.3% is within expected 45-75% range |
| Model beats baseline | ✅ Passed | +29.0% improvement over baseline |

### 6.2 Verification Notebook

Run `notebooks/leakage_check.ipynb` to independently verify all checks.

---

## 7. Conclusions

### 7.1 What We Learned

1. **Data leakage is subtle and devastating** — 87-92% accuracy looked impressive but was meaningless
2. **Temporal validation is essential** for time-series prediction problems
3. **Lower accuracy can mean better science** — honest 57.3% beats inflated 87-92%
4. **2023 predictions are now possible** — the original project couldn't make true predictions
5. **All leakage checks verified** — 7 independent checks confirm data integrity

### 7.2 Practical Implications

For NCAA athletic administrators:
- Model identifies schools with high probability of financial trajectory change
- Confidence scores help prioritize interventions
- 45 institutions have high-confidence (≥70%) predictions for 2023
- Predictions should inform, not dictate, decisions
- 57.3% accuracy means significant uncertainty remains
- Best model (Logistic Regression) provides +29% improvement over baseline

### 7.3 Future Work

1. **Feature Engineering**: Explore additional predictive signals
2. **Model Tuning**: More extensive hyperparameter optimization
3. **Ensemble Methods**: Combine model predictions
4. **Validation**: Compare 2023 predictions to 2024 actuals when available

---

## Appendix A: Notebook Flow

```
00_Data_Loading_and_Overview.ipynb
    ↓ (understanding data structure)
01_Exploratory_Data_Analysis.ipynb
    ↓ (discovering patterns)
02_Feature_Discovery.ipynb
    ↓ (proposing features)
03_Target_Definition.ipynb
    ↓ (defining classes)
04_Feature_Engineering.ipynb
    ↓ (building features)
05_Temporal_Split.ipynb
    ↓ (splitting by time)
06_Model_Training.ipynb
    ↓ (training models)
07_Model_Evaluation.ipynb
    ↓ (honest evaluation)
08_Predictions.ipynb
    ↓ (2023 predictions)
leakage_check.ipynb (verification)
```

---

## Appendix B: Files Generated

| File | Purpose |
|------|---------|
| `data/processed/features_temporal.csv` | All features and targets |
| `data/processed/train.csv` | Training data (2017-2019) |
| `data/processed/val.csv` | Validation data (2020-2021) |
| `data/processed/test.csv` | Test data (2022) |
| `data/processed/features_2023_predict.csv` | 2023 prediction data |
| `models/*.pkl` | Trained models and preprocessing artifacts |
| `reports/evaluation_results.csv` | Model performance metrics |
| `reports/predictions_2023.csv` | All 2023 predictions |
| `reports/predictions_2023_high_confidence.csv` | High-confidence predictions |

---

*Report generated as part of the Temporal_Validated project rebuild.*
*Last updated: November 28, 2025*
*All leakage checks verified: ✅ PASSED*
