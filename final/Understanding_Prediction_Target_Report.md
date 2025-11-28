# Understanding the Prediction Target: What Are We Actually Predicting?

## The Fundamental Question

> "We make predictions like 'Improving', 'Declining', 'Stable' — what does that actually mean? Can't I just compare ratios? Isn't this just basic math?"

This document explains **why this project uses Machine Learning** and **what the target variable actually represents**.

---

## Table of Contents

1. [The Common Misconception](#1-the-common-misconception)
2. [What the Target Actually Means](#2-what-the-target-actually-means)
3. [Why Simple Math Doesn't Work](#3-why-simple-math-doesnt-work)
4. [The Real ML Problem](#4-the-real-ml-problem)
5. [UT Austin Example: A Deep Dive](#5-ut-austin-example-a-deep-dive)
6. [Why 52 Features Instead of Just Ratio?](#6-why-52-features-instead-of-just-ratio)
7. [Summary: The One-Liner Explanation](#7-summary-the-one-liner-explanation)

---
You 
## 1. The Common Misconception

### What You Might Think

```
"Improving" means 2022 improved compared to 2021
→ Just compare: ratio_2022 vs ratio_2021
→ If ratio went up, it's "Improving"
→ This is basic math, not ML!
```

### The Reality

```
"Improving" means: From 2022 → 2023, the school WILL improve
→ This is a PREDICTION about the FUTURE
→ We don't have 2023 data yet!
→ We need ML to predict unseen data
```

| What You Thought | What's Actually Happening |
|------------------|---------------------------|
| Target = "Did 2022 improve vs 2021?" | Target = "What will happen from 2022 → 2023?" |
| Based on **PAST** data | Based on **FUTURE** outcomes |
| Simple ratio comparison | Complex pattern recognition |
| No ML needed | ML is essential |

---

## 2. What the Target Actually Means

### The Target Definition (from the code)

```python
def define_trajectory(row):
    rev_growth = row['Future_Revenue_Growth']  # What happens NEXT year
    exp_growth = row['Future_Expense_Growth']  # What happens NEXT year
    
    # Improving: Revenue grows > 3% AND expenses grow LESS than revenue
    if rev_growth > 0.03 and exp_growth < rev_growth:
        return "Improving"
    
    # Declining: Revenue shrinks OR expenses grow much faster than revenue  
    if rev_growth < 0 or exp_growth > rev_growth + 0.03:
        return "Declining"
    
    # Stable: Everything else
    return "Stable"
```

### Key Insight

The target for **year T** is calculated using data from **year T+1**.

```
Target for 2022 = What happens between 2022 → 2023
                = Based on 2023's revenue and expense growth
                = We need to PREDICT this (we don't have 2023 data)
```

---

## 3. Why Simple Math Doesn't Work

### The Simple Math Approach (WRONG)

```python
# You might think:
ratio_2022 = revenue_2022 / expense_2022  # = 1.311
ratio_2021 = revenue_2021 / expense_2021  # = 1.196

if ratio_2022 > ratio_2021:
    trajectory = "Improving"  # Simple comparison!
```

### Testing This on UT Austin

| Year | Ratio | Δ Ratio | Simple Math Says | Actual Target | Match? |
|------|-------|---------|------------------|---------------|--------|
| 2018 | 1.232 | +0.091 | Improving | **Declining** | ❌ |
| 2019 | 1.283 | +0.050 | Improving | **Declining** | ❌ |
| 2020 | 1.000 | -0.283 | Declining | **Improving** | ❌ |
| 2021 | 1.196 | +0.196 | Improving | Improving | ✅ |
| 2022 | 1.311 | +0.116 | Improving | Improving | ✅ |

**Simple ratio comparison is WRONG 3 out of 5 times!**

### Why the Mismatch?

The target isn't about **past performance** — it's about **future trajectory**.

Example: **2020**
- Ratio DROPPED from 1.283 to 1.000 (COVID crash)
- Simple math says: "Declining" ❌
- But from 2020 → 2021, revenue grew 53.4% with expense growth of 28.3%
- Actual target: "Improving" ✅ (great recovery ahead)

---

## 4. The Real ML Problem

### The Challenge

```
Given: 2022's features (ratio, momentum, growth rates, volatility, etc.)
Predict: Will the trajectory from 2022 → 2023 be Improving, Declining, or Stable?

Constraint: We DON'T have 2023 data!
```

### Why This Needs ML

1. **We're predicting the FUTURE**, not classifying the past
2. **52 features** interact in complex, non-linear ways
3. **Patterns aren't obvious** — high ratio doesn't guarantee improvement
4. **Historical context matters** — momentum, volatility, trends

### The ML Solution

```
1. Training Phase (2017-2021):
   - For each year, we KNOW what happened next
   - Model learns: "When features look like X, next year is Improving"

2. Prediction Phase (2022):
   - Feed 2022's features to the model
   - Model outputs: Predicted trajectory for 2023
   - This is a TRUE prediction (unverified until 2024 data arrives)
```

---

## 5. UT Austin Example: A Deep Dive

### Year-by-Year Financial Journey

| Year | Revenue | Expenses | What Happens Next | Target |
|------|---------|----------|-------------------|--------|
| 2017 | $210M | $184M | Rev +2.6%, Exp -5.0% | Stable |
| 2018 | $216M | $175M | Rev -11.2%, Exp -14.7% | **Declining** |
| 2019 | $192M | $149M | Rev -21.6%, Exp +0.5% | **Declining** |
| 2020 | $150M | $150M | Rev +53.4%, Exp +28.3% | **Improving** |
| 2021 | $231M | $193M | Rev +13.4%, Exp +3.4% | **Improving** |
| 2022 | $261M | $199M | Rev +22.6%, Exp +19.2% | **Improving** |

### Breaking Down 2018 (Counter-Intuitive Case)

```
2018 Data:
- Ratio: 1.232 (GREAT! 23% surplus)
- Ratio improved from 2017's 1.141

Simple math says: "Improving" ✅

But what ACTUALLY happened (2018 → 2019):
- Revenue: -11.2% (SHRINKING!)
- This triggers the Declining rule

Actual target: "Declining" ❌

The model needs to predict this DECLINE despite the current good ratio!
```

### Breaking Down 2020 (COVID Case)

```
2020 Data:
- Ratio: 1.000 (break-even, worst in years)
- Ratio CRASHED from 2019's 1.283

Simple math says: "Declining" ❌

But what ACTUALLY happened (2020 → 2021):
- Revenue: +53.4% (MASSIVE recovery!)
- Expense: +28.3% (grew less than revenue)
- This triggers the Improving rule

Actual target: "Improving" ✅

The model needs to predict this RECOVERY despite the current crash!
```

---

## 6. Why 52 Features Instead of Just Ratio?

### The Target Rules Require Future Data

```python
# To know if 2022 is "Improving", we need:
revenue_growth_2023 = (revenue_2023 / revenue_2022) - 1  # Unknown!
expense_growth_2023 = (expense_2023 / expense_2022) - 1  # Unknown!
```

We don't have 2023 data, so we use **predictive features**:

### The 52 Features (Grouped)

| Category | Features | Why They Help |
|----------|----------|---------------|
| **Current State** | Revenue, Expenses, Ratio | Baseline financial health |
| **Momentum** | Efficiency_Momentum, Revenue_Growth_Momentum | Direction of change |
| **Lag Features** | Lag1_Ratio, Lag1_Growth, Lag1_Target | What happened last year |
| **Volatility** | Revenue_Volatility, Expense_Volatility | Stability patterns |
| **Averages** | 2-year means, CAGRs | Smoothed trends |
| **Per-Athlete** | Revenue_Per_Athlete, Expense_Per_Athlete | Efficiency metrics |
| **Categorical** | Division (D1/D2/D3) | Program tier effects |

### The Key Insight

> **Current ratio tells you WHERE you are.**
> **Momentum and trends tell you WHERE you're going.**

The model learns: "Schools with positive momentum and controlled expense growth tend to KEEP improving."

---

## 7. Summary: The One-Liner Explanation

### For a Quick Answer

> **"We're not classifying what already happened — we're predicting what will happen next year, using patterns learned from historical data."**

### The Full Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE ML PREDICTION PROBLEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   INPUT: 2022 Features (52 variables)                           │
│   ├── Current ratio: 1.311                                      │
│   ├── Efficiency momentum: +0.156                               │
│   ├── Revenue growth: +13.4%                                    │
│   ├── Expense growth: +3.4%                                     │
│   └── ... (48 more features)                                    │
│                                                                 │
│   OUTPUT: Predicted Trajectory for 2022 → 2023                  │
│   └── "Improving" with 99.91% confidence                        │
│                                                                 │
│   GROUND TRUTH: We can verify because we have 2023 data         │
│   └── 2023: Rev +22.6%, Exp +19.2% → Actually "Improving" ✅    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   WHY ML?                                                       │
│   • Can't calculate future growth without future data           │
│   • 52 features interact in complex ways                        │
│   • Model learns patterns humans can't easily see               │
│   • 87.6% accuracy proves the patterns exist and are learnable  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Common Follow-Up Questions

### Q: "So the 2022 predictions are verifiable?"

**A:** Yes! For 2022, we have 2023 data, so we can check:
- Model predicted: "Improving"
- Actual 2022→2023: Rev +22.6%, Exp +19.2% → "Improving" ✅

### Q: "What about predicting 2024?"

**A:** If we had 2024 features, we could predict 2024→2025 trajectory. But we wouldn't be able to VERIFY it until 2025 data comes out.

### Q: "Why not just use the growth rate columns?"

**A:** Those are **past** growth rates (2021→2022). The target needs **future** growth rates (2022→2023), which we predict.

### Q: "Is 87.6% accuracy good?"

**A:** Yes! Random guessing would give ~33% (3 classes). We're nearly 3x better than random, proving the model learned real patterns.

---

## Document Information

- **Created:** November 27, 2025
- **Purpose:** Clarify the fundamental ML problem and target definition
- **Context:** Questions raised during project review session
- **Related Files:** 
  - `final/Report.md` (main project report)
  - `archive/today/notebooks/04_Trajectory_Feature_Engineering.ipynb` (target definition)
  - `final/assets/scripts/predict.py` (prediction script)

---

*"The confusion was thinking we classify the PAST. We're actually predicting the FUTURE."*
