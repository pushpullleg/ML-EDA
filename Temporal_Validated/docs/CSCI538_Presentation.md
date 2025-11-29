# CSCI 538 Final Project Presentation

# Predicting NCAA Athletic Department Financial Trajectories: A Machine Learning Approach

**Mukesh Ravichandran & Abner Lusung**
**East Texas A&M University**
**December 2025**

---

## Presentation Outline (10-15 slides, ~15-20 minutes)

---

### Slide 1: Title Slide

**Temporal Validation in NCAA Athletics Financial Trajectory Prediction: Addressing Data Leakage in Machine Learning Models**

- Mukesh Ravichandran
- Abner Lusung
- CSCI 538 Final Project
- East Texas A&M University
- December 2025

---

### Slide 2: Problem Statement

**The Challenge**
- Predicting financial trajectory of 1,700+ NCAA athletic programs
- Critical for resource allocation and institutional planning
- Three-class prediction: Improving, Stable, Declining

**The Risk: Data Leakage**
- Common pitfall in time series ML (Kaufman et al., 2012)
- Can produce artificially high accuracy that fails in production

**Our Goal**
- Build prediction system with proper temporal validation
- Generate honest, actionable predictions

---

### Slide 3: What is Data Leakage?

**Definition**: When information from outside the training dataset is used to create the model

**Two Common Sources of Leakage:**

1. **Target-Derived Features**
   - Using lagged target labels as features
   - Model gets encoded answers as input!

2. **Random Train-Test Split**
   - Future data in training, past data in testing
   - Model "predicts" the past using future knowledge

**Our Solution**: Strict temporal validation to prevent both issues

---

### Slide 4: Data Overview

**Source**: NCAA EADA (Equity in Athletics Disclosure Act)

| Metric | Value |
|--------|-------|
| Total Records | 17,220 |
| Institutions | 1,722 |
| Time Span | 2014-2023 (10 years) |
| Original Features | 580 |
| Engineered Features | 14 |

**Target Variable**: Financial Trajectory
- Improving (revenue up, expenses controlled)
- Stable (no significant change)
- Declining (revenue down or expenses outpacing)

---

### Slide 5: Temporal Validation Methodology

**The Fix**: Split data by TIME, not randomly

```
2014-2016     2017-2019     2020-2021     2022         2023
   |             |             |            |            |
   v             v             v            v            v
[LOOKBACK]  [TRAINING]   [VALIDATION]  [TEST]    [PREDICTION]
(for lags)   Learn         Tune       Final      True future
             patterns      params     eval       predictions
```

**Key Principle**: Never let the model see the future during training

---

### Slide 6: Feature Engineering

**14 Leak-Free Features** (using only past data)

| Category | Features |
|----------|----------|
| Current Metrics | Efficiency_Ratio, Revenue_Per_Athlete, Total_Athletes |
| 1-Year Growth | Revenue_Growth_1yr, Expense_Growth_1yr |
| 2-Year Trends | Revenue_CAGR_2yr, Expense_CAGR_2yr, Efficiency_Mean_2yr |
| Volatility | Revenue_Volatility_2yr, Expense_Volatility_2yr |
| Raw Values | Grand Total Revenue, Grand Total Expenses |
| Categorical | Division (D1/D2/D3/Other) |

**Forbidden**: Any feature containing "Target", "Label", or "Future_"

---

### Slide 7: Model Selection

**Three Algorithms Tested:**

| Model | Key Settings |
|-------|-------------|
| Logistic Regression | Balanced weights, L-BFGS solver |
| Random Forest | 100 trees, max_depth=10, balanced |
| XGBoost | 100 estimators, max_depth=5 |

**Class Imbalance Handling:**
- SMOTE (Synthetic Minority Over-sampling)
- Balanced class weights

**Why Traditional ML vs Deep Learning?**
- ~5,000 training samples (too small for deep learning)
- Interpretability (feature importance)
- Computational efficiency

---

### Slide 8: Results - Model Performance

**2022 Holdout Test Results**

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
|-------|----------|---------------|------------|
| **Logistic Regression** | **57.3%** | 0.563 | 0.487 |
| Random Forest | 54.6% | 0.560 | 0.498 |
| XGBoost | 53.7% | 0.551 | 0.491 |

**Best Model**: Logistic Regression (57.3% accuracy)

---

### Slide 9: Results - Baseline Comparison

**Is 57.3% Good?**

| Approach | Accuracy | vs Baseline |
|----------|----------|-------------|
| Random Guessing | 33.3% | — |
| Most-Frequent Class | 28.3% | — |
| **Our Best Model** | **57.3%** | **+29%** |

**Key Insight**: 
- 29 percentage points above random baseline
- Genuine predictive value for a difficult problem
- Honest accuracy that stakeholders can trust

---

### Slide 10: Results - Visualizations

*[Include these figures from reports/ folder]*

- **Confusion Matrices**: Per-model performance breakdown
- **Model Comparison Chart**: Side-by-side accuracy comparison
- **Feature Importance**: Top predictive features
- **Temporal Split Diagram**: How data was divided

---

### Slide 11: 2023 Predictions

**Predictions for 1,722 Institutions**

| Trajectory | Count | Percentage |
|------------|-------|------------|
| Stable | 803 | 46.6% |
| Declining | 473 | 27.5% |
| Improving | 446 | 25.9% |

**High-Confidence Cases (≥70%)**: 45 institutions
- Thomas University: 90% declining confidence
- University of Olivet: 89% declining confidence

**These are TRUE predictions** - answers don't exist yet!

---

### Slide 12: Key Findings

1. **Data leakage prevention is critical**
   - Proper methodology ensures reliable results

2. **Temporal validation is essential**
   - For any time series problem, split by time

3. **57.3% accuracy has real value**
   - 29% improvement over baseline
   - Stakeholders can trust these predictions

4. **2023 predictions are actionable**
   - 1,722 institutions with trajectory forecasts
   - 45 high-confidence cases for priority attention

5. **All 7 verification checks passed**
   - Rigorous methodology validation

---

### Slide 13: Limitations and Future Work

**Limitations:**
- External factors not modeled (economy, policy changes)
- COVID-19 impact on 2020-2021 data
- Self-reported data quality
- 57.3% accuracy = significant uncertainty remains

**Future Work:**
- Add macroeconomic indicators
- Explore LSTM/Transformer models
- Build ensemble voting system
- Validate 2023 predictions with 2024 data
- Develop real-time prediction API

---

### Slide 14: Conclusions

**What We Accomplished:**
- Built prediction system with proper temporal validation
- Engineered 14 leak-free features
- Achieved 57.3% accuracy (+29% over baseline)
- Generated actionable predictions for 1,722 institutions
- Verified methodology through 7 independent checks

**Main Takeaway:**
> "Proper temporal validation ensures predictions that stakeholders can trust for real decision-making."

**For Athletic Administrators:**
- Early warning system for financial decline
- Decision support tool (not replacement)
- High-confidence cases (45 institutions) warrant priority attention

---

### Slide 15: References & Q&A

**Key References:**
- Kaufman et al. (2012) - Data leakage in ML
- Bergmeir & Benítez (2012) - Time series cross-validation
- Breiman (2001) - Random Forests
- Chen & Guestrin (2016) - XGBoost
- Chawla et al. (2002) - SMOTE
- NCAA EADA Database

**Questions?**

*Thank you for your attention!*

---

## Speaker Notes

### Role Division for Presentation

**Mukesh Ravichandran (Slides 1-4, 8-10, 14-15):**
- Introduction and problem statement
- Data leakage explanation
- Results and visualizations
- Conclusions and Q&A

**Abner Lusung (Slides 5-7, 11-13):**
- Temporal validation methodology
- Feature engineering
- Model selection
- 2023 predictions
- Limitations and future work

### Timing Guide
- Introduction (Slides 1-4): ~4 minutes
- Methodology (Slides 5-7): ~4 minutes
- Results (Slides 8-11): ~5 minutes
- Discussion (Slides 12-14): ~4 minutes
- Q&A (Slide 15): ~3 minutes
- **Total**: ~20 minutes

### Key Points to Emphasize
1. Temporal validation as non-negotiable for time series ML
2. 57.3% accuracy = 29% improvement over random baseline
3. Real predictions for 2023 ready for stakeholder use
4. Practical value for NCAA administrators
5. Rigorous methodology verified through 7 checks

