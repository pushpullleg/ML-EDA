# Prediction Case Study: Texas A&M University-Commerce

## 1. Prediction Result

```
School:              Texas A & M University-Commerce
Year of Data:        2022 (most recent available)
Predicted Trajectory: IMPROVING ✅
Confidence:          Declining 42.24% | Stable 0.24% | Improving 57.51%
```

---v

## 2. How Was This Prediction Made?

### Step 1: Input Data
The model used 46 features from the school's 2022 financial data, including:

| Feature | Value | Meaning |
|---------|-------|---------|
| Division | D1 | NCAA Division I school |
| Total Athletes | 285 | Number of student athletes |
| Grand Total Revenue | $15,568,324 | Total athletic department revenue |
| Grand Total Expenses | $15,344,594 | Total athletic department expenses |
| Revenue/Expense Ratio | 1.015 | Revenue slightly exceeds expenses (healthy) |
| Revenue Growth (1yr) | +2.94% | Revenue increased from last year |
| Expense Growth (1yr) | +2.91% | Expenses increased similarly |
| Efficiency Momentum | +0.0038 | Efficiency is slightly improving |
| Revenue Per Athlete | $54,626 | Revenue generated per athlete |
| Expense Per Athlete | $53,841 | Cost per athlete |

### Step 2: Model Processing
1. **Preprocessing**: Categorical features (Division) were one-hot encoded
2. **XGBoost**: The trained model evaluated all 46 features
3. **Output**: Probability distribution across 3 classes

### Step 3: Prediction Logic
The model outputs probabilities for each class:
- **Declining**: 42.24%
- **Stable**: 0.24%
- **Improving**: 57.51% ← Highest probability = Final prediction

---

## 3. What Does This Prediction Mean?

### Interpretation
Texas A&M University-Commerce is predicted to be on an **Improving** financial trajectory. This means:
- Their revenue-to-expense ratio is expected to **increase** over the next year
- Financial health is trending **positive**

### Confidence Level
The prediction is a **close call** (57% vs 42% for Declining). This suggests:
- The school is at a **crossroads** financially
- Small changes in revenue or expenses could flip the trajectory
- Management should **monitor closely** and maintain current positive trends

### Why Improving?
Based on the data:
1. **Revenue > Expenses**: Ratio of 1.015 means they're slightly profitable
2. **Positive Momentum**: Efficiency has been improving (+0.0038)
3. **Historical Pattern**: 3 of last 6 years were "Improving"
4. **Balanced Growth**: Revenue and expense growth are nearly equal (~3%)

---

## 4. Historical Trajectory

| Year | Trajectory | Trend |
|------|------------|-------|
| 2017 | Stable | — |
| 2018 | Stable | — |
| 2019 | Improving | ↑ |
| 2020 | Improving | ↑ |
| 2021 | Stable | — |
| 2022 | Improving | ↑ |

**Pattern**: The school alternates between Stable and Improving — they've never been classified as Declining in recent years.

---

## 5. Is This School in the Final Dataset?

**Yes.** Texas A&M University-Commerce is included in the final dataset (`trajectory_excellent.csv`).

| Dataset | Rows | Status |
|---------|------|--------|
| Raw EADA Data | 10 years (2014-2023) | ✅ Present |
| Final Dataset | 6 years (2017-2022) | ✅ Present |

**Why only 6 years in final?**
- The first 2 years are used to compute lag features (Year t-1, t-2)
- 2023 data is held out for future prediction
- This is a data engineering requirement, not data removal

---

## 6. Command Used

```bash
.venv/bin/python final/assets/scripts/predict.py "Texas A & M University-Commerce"
```

**Note**: The exact name format matters — use spaces around the `&` symbol.

---

## 7. Actionable Insights for Management

| Insight | Recommendation |
|---------|----------------|
| Revenue slightly exceeds expenses | Maintain current revenue streams |
| Efficiency momentum is positive | Continue cost discipline |
| Close call (57% Improving) | Don't become complacent — monitor monthly |
| Historical pattern is favorable | Leverage positive trend for stakeholder confidence |

---

*Generated: November 27, 2025*
*Model: XGBoost + SMOTE (87.6% accuracy, 0.968 ROC-AUC)*
