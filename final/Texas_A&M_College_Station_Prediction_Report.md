# Prediction Case Study: Texas A&M University-College Station

## 1. Prediction Result

```
School:              Texas A & M University-College Station
Year of Data:        2022 (most recent available)
Predicted Trajectory: DECLINING 🔴v
Confidence:          Declining 77.02% | Stable 22.75% | Improving 0.23%
```

---

## 2. Why This School Is Interesting

Despite having a **healthy surplus** (11% more revenue than expenses), the model predicts **DECLINING**. This demonstrates that the model looks at **trends**, not just current profitability.

---

## 3. Historical Financial Journey

| Year | Revenue | Expenses | Ratio | Δ Ratio | Trend |
|------|---------|----------|-------|---------|-------|
| 2017 | $152,971,142 | $143,231,483 | **1.068** | — | Baseline |
| 2018 | $160,101,611 | $145,678,287 | **1.099** | +0.031 🟢 | Improved |
| 2019 | $143,807,835 | $136,327,986 | **1.055** | -0.044 🔴 | Declined |
| 2020 | $136,858,666 | $121,604,428 | **1.125** | +0.070 🟢 | Improved |
| 2021 | $169,220,001 | $157,702,310 | **1.073** | -0.052 🔴 | Declined |
| 2022 | $194,254,820 | $174,723,140 | **1.112** | +0.039 🟢 | Improved |

---

## 4. The Paradox: Why "Declining" With a Surplus?

Looking at 2022 alone, Texas A&M looks great:
- ✅ Revenue **$19.5M higher** than expenses
- ✅ Ratio of 1.112 (11% surplus)
- ✅ Revenue grew 14.79%

**But the model sees something concerning:**

### Efficiency Momentum = -0.0068 (Negative!)

Despite the current surplus, the **trend over time** shows volatility:
```
Pattern: Up → Down → Up → Down → Up → ???
         1.07 → 1.10 → 1.05 → 1.13 → 1.07 → 1.11

The ratio keeps swinging between ~1.05 and ~1.12
Model detects: Unstable efficiency pattern
```

### Expense Growth Catching Up
```
Revenue Growth:  +14.79%
Expense Growth:  +10.79%
Gap:             Only 4% difference

Trend: Expenses are growing almost as fast as revenue
```

---

## 5. Key Features (2022 Data)

| Feature | Value | Analysis |
|---------|-------|----------|
| Division | D1 | SEC powerhouse |
| Total Athletes | 620 | Large program |
| Grand Total Revenue | $194,254,820 | ~$194M budget |
| Grand Total Expenses | $174,723,140 | ~$175M expenses |
| Revenue/Expense Ratio | 1.112 | 11% surplus ✅ |
| Efficiency Mean (2yr) | 1.092 | Good average |
| **Efficiency Momentum** | **-0.007** | ⚠️ **Negative trend** |
| Revenue Growth (1yr) | +14.79% | Strong growth |
| Expense Growth (1yr) | +10.79% | ⚠️ Fast expense growth |
| Revenue Per Athlete | $313,314 | High revenue per athlete |
| Expense Per Athlete | $281,812 | But costs also high |

---

## 6. Why the Model Predicts "DECLINING"

The 77% confidence in "Declining" comes from:

1. **Negative Momentum (-0.007)**: Despite current surplus, the trajectory trend is downward
2. **Volatility Pattern**: Alternating up/down years suggest instability
3. **Expense Growth Rate**: At 10.79%, expenses are chasing revenue
4. **Historical Declines**: Already had declining years in 2019 and 2021

### The Key Insight
> **Current surplus ≠ Future stability**
> 
> The model learned that schools with oscillating ratios often trend downward. A&M's pattern of "up-down-up-down" signals risk.

---

## 7. Comparison: Texas A&M vs Its Rivals

| Metric | A&M College Station | UT Austin | Alabama |
|--------|---------------------|-----------|---------|
| 2022 Ratio | 1.112 | 1.311 | 1.000 |
| Efficiency Momentum | -0.007 | **+0.156** | -0.033 |
| Revenue Growth | 14.79% | 13.38% | -1.02% |
| Expense Growth | 10.79% | **3.39%** | 9.43% |
| Prediction | 🔴 Declining | 🟢 Improving | 🟡 Stable |

**Key Difference**: UT Austin controls expenses better (only 3.39% growth vs A&M's 10.79%), giving them stronger momentum.

---

## 8. Actionable Insights for A&M's Management

| Observation | Recommendation |
|-------------|----------------|
| Expense growth (10.79%) nearly matches revenue (14.79%) | 🔴 **Slow expense growth** — target <5% |
| Ratio oscillates yearly | Implement multi-year budgeting for stability |
| Currently at 1.11 ratio | Build reserve fund during surplus years |
| Revenue per athlete is high ($313K) | Maintain revenue streams during SEC transition |
| Negative efficiency momentum | Reverse trend before it compounds |

---

## 9. What This Means for Your Presentation

This is a **perfect interview talking point**:

> "Even schools with current surpluses can be predicted as 'Declining' based on momentum and volatility patterns. Texas A&M has an 11% surplus today, but the model gives it 77% chance of declining because expenses are growing fast and efficiency fluctuates year-to-year."

---

## 10. Command Used

```bash
.venv/bin/python final/assets/scripts/predict.py "Texas A & M University-College Station"
```

---

*Generated: November 27, 2025*
*Model: XGBoost + SMOTE (87.6% accuracy, 0.968 ROC-AUC)*
