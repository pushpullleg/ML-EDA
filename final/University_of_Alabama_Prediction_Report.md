# Prediction Case Study: The University of Alabama

## 1. Prediction Result

```
School:              The University of Alabama
Year of Data:        2022 (most recent available)
Predicted Trajectory: STABLE 🟡
Confidence:          Declining 0.14% | Stable 71.33% | Improving 28.53%
```

---

## 2. Why This School Is Interesting

Unlike Texas A&M Commerce (which was consistently stable/improving), **Alabama shows dramatic fluctuations** in efficiency — making it an excellent example of what "Declining" actually looks like.

---

## 3. Historical Financial Journey

| Year | Revenue | Expenses | Ratio | Δ Ratio | What Happened |
|------|---------|----------|-------|---------|---------------|
| 2017 | $181,470,156 | $149,583,715 | **1.213** | — | Strong surplus (21% more revenue than expenses) |
| 2018 | $166,812,799 | $166,812,799 | **1.000** | **-0.213** 🔴 | **CRASHED** to break-even |
| 2019 | $189,242,298 | $149,767,295 | **1.264** | +0.264 🟢 | Recovered strongly |
| 2020 | $157,062,672 | $147,454,823 | **1.065** | **-0.198** 🔴 | **DROPPED** again |
| 2021 | $193,168,171 | $174,715,501 | **1.106** | +0.041 🟢 | Improved |
| 2022 | $191,190,486 | $191,190,486 | **1.000** | **-0.106** 🔴 | Back to break-even |

---

## 4. Understanding the DECLINING Years

### 2017 → 2018: Major Decline
```
2017: Revenue $181M, Expenses $150M → Ratio 1.213 (keeping $32M surplus)
2018: Revenue $167M, Expenses $167M → Ratio 1.000 (zero surplus)

What happened:
- Revenue DROPPED by $15M
- Expenses INCREASED by $17M
- Result: Lost their entire $32M cushion
- Trajectory: 🔴 DECLINING
```

### 2019 → 2020: Another Decline
```
2019: Ratio 1.264 (very healthy, 26% surplus)
2020: Ratio 1.065 (only 6.5% surplus)

What happened:
- Revenue dropped $32M (COVID impact?)
- Ratio fell by 0.20
- Trajectory: 🔴 DECLINING
```

### 2021 → 2022: Current Decline
```
2021: Ratio 1.106 (10.6% surplus)
2022: Ratio 1.000 (break-even)

What happened:
- Revenue stayed flat (-1%)
- Expenses jumped +9.4%
- Lost their entire surplus
- Efficiency Momentum: -0.0326 (negative = declining trend)
```

---

## 5. Key Features (2022 Data)

| Feature | Value | Meaning |
|---------|-------|---------|
| Division | D1 | NCAA Division I (Power conference) |
| Total Athletes | 641 | Large athletic program |
| Grand Total Revenue | $191,190,486 | ~$191M budget |
| Grand Total Expenses | $191,190,486 | Exactly equals revenue |
| Revenue/Expense Ratio | 1.000 | Break-even (no surplus) |
| Efficiency Mean (2yr) | 1.053 | Average of last 2 years still positive |
| **Efficiency Momentum** | **-0.033** | ⚠️ **Trending downward** |
| Revenue Growth (1yr) | -1.02% | Revenue slightly decreased |
| Expense Growth (1yr) | +9.43% | ⚠️ Expenses growing fast |
| Revenue Per Athlete | $298,269 | High revenue per athlete |
| Expense Per Athlete | $298,269 | But equally high costs |

---

## 6. Why the Model Predicts "STABLE" (Not Declining)

Despite the negative trends, the model predicts **Stable** with 71% confidence because:

1. **Historical Pattern**: Alabama has bounced back before (2018→2019 recovery, 2020→2021 recovery)
2. **2-Year Average Still Positive**: Efficiency Mean of 1.053 suggests underlying health
3. **Size Advantage**: Large programs have more levers to pull for recovery
4. **Revenue Per Athlete**: Still generating nearly $300K per athlete

However, note the **28.5% chance of Improving** and only **0.14% chance of Declining** — the model sees potential for recovery, not continued decline.

---

## 7. Contrast with Texas A&M Commerce

| Metric | Alabama | Texas A&M Commerce |
|--------|---------|-------------------|
| Budget Size | $191M | $15.6M |
| Ratio Volatility | High (0.00 to 1.26) | Low (1.00 to 1.015) |
| Efficiency Momentum | -0.033 (negative) | +0.004 (positive) |
| Historical Declines | 3 times | 0 times |
| Prediction | Stable (71%) | Improving (57%) |

**Key Insight**: Alabama is a "volatile" program — big swings between surplus and break-even. Commerce is a "steady" program — consistent small improvements.

---

## 8. Actionable Insights for Alabama's Management

| Observation | Recommendation |
|-------------|----------------|
| Expenses grew 9.4% while revenue dropped 1% | 🔴 **Immediate cost review needed** |
| Currently at break-even (ratio 1.0) | No margin for error — any expense increase creates deficit |
| Efficiency momentum is negative | Trend must be reversed within 1-2 budget cycles |
| Historical pattern shows recovery ability | Leverage past strategies that restored surplus |
| Revenue per athlete is high | Focus on maintaining revenue streams (ticket sales, TV deals) |

---

## 9. Command Used

```bash
.venv/bin/python final/assets/scripts/predict.py "The University of Alabama"
```

---

*Generated: November 27, 2025*
*Model: XGBoost + SMOTE (87.6% accuracy, 0.968 ROC-AUC)*
