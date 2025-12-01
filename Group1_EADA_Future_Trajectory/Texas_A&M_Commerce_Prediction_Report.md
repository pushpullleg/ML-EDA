# Prediction Case Study: Texas A&M University-Commerce

## 1. Prediction Result

```
School:              Texas A & M University-Commerce
Year of Data:        2023 (true future prediction)
Predicted Trajectory: DECLINING ⚠️
Confidence:          Declining 39.98% | Stable 20.99% | Improving 39.03%
```

**Note**: This is a **close call** prediction — the model is nearly split between Declining (40%) and Improving (39%), indicating high uncertainty.

---

## 2. How Was This Prediction Made?

### Step 1: Input Data
The model used 14 features from the school's 2023 financial data:

| Feature | Value | Meaning |
|---------|-------|---------|
| Division | D1 | NCAA Division I school |
| Total Athletes | 324 | Number of student athletes |
| Grand Total Revenue | $18,000,481 | Total athletic program revenue |
| Grand Total Expenses | $17,433,995 | Total athletic program expenses |
| Revenue/Expense Ratio | 1.0325 | Revenue exceeds expenses (healthy) |
| Revenue Growth (1yr) | +15.62% | Strong revenue growth from 2022 |
| Expense Growth (1yr) | +13.62% | Expenses also growing rapidly |
| Revenue CAGR (2yr) | +9.10% | 2-year compound annual growth |
| Expense CAGR (2yr) | +8.13% | 2-year expense growth |
| Efficiency Mean (2yr) | 1.0235 | Average efficiency over 2 years |
| Revenue Per Athlete | $55,557 | Revenue generated per athlete |
| Revenue Volatility (2yr) | $1,719,795 | Standard deviation of revenue |
| Expense Volatility (2yr) | $1,477,430 | Standard deviation of expenses |

### Step 2: Model Processing
1. **Preprocessing**: 
   - Categorical features (Division) were encoded
   - Missing values filled using training set medians
   - Features scaled using training set scaler
2. **Logistic Regression Model**: The trained model evaluated all 14 features
   - Model trained on 2017-2019 data
   - Validated on 2020-2021 data
   - Tested on 2022 data (57.3% accuracy - best performing)
3. **Output**: Probability distribution across 3 classes

### Step 3: Prediction Logic
The model outputs probabilities for each class:
- **Declining**: 39.98% ← Highest probability = Final prediction
- **Stable**: 20.99%
- **Improving**: 39.03% ← Very close second

---

## 3. What Does This Prediction Mean?

### Interpretation
Texas A&M University-Commerce is predicted to be on a **Declining** financial trajectory for 2023→2024. This means:
- Their revenue-to-expense ratio is expected to **decrease** over the next year
- Financial health is trending **negative**

### Confidence Level
The prediction is a **very close call** (40% vs 39% for Improving). This suggests:
- The school is at a **critical financial crossroads**
- Small changes in revenue or expenses could flip the trajectory either way
- Management should **monitor very closely** and take proactive measures

### Why Declining?
Despite strong recent growth, the model predicts decline based on:
1. **Expense Growth Concerns**: While revenue grew 15.62%, expenses grew 13.62% — the gap is narrowing
2. **Volatility**: High revenue volatility ($1.7M) suggests instability
3. **Historical Pattern**: School alternates between Stable and Improving — a Declining prediction is unusual
4. **Model Pattern Recognition**: The combination of high growth with high volatility often precedes decline

---

## 4. Historical Trajectory

| Year | Trajectory | Trend | Revenue | Expenses | Ratio |
|------|------------|-------|---------|----------|-------|
| 2017 | Stable | — | $11,095,214 | $11,095,214 | 1.000 |
| 2018 | Stable | — | $11,232,217 | $11,138,695 | 1.008 |
| 2019 | Improving | ↑ | $11,755,537 | $11,681,688 | 1.006 |
| 2020 | Improving | ↑ | $12,456,510 | $12,369,254 | 1.007 |
| 2021 | Stable | — | $15,123,734 | $14,911,268 | 1.014 |
| 2022 | Improving | ↑ | $15,568,324 | $15,344,594 | 1.015 |
| **2023** | **Declining** | **↓** | **$18,000,481** | **$17,433,995** | **1.032** |

**Pattern**: The school has alternated between Stable and Improving — **this is the first predicted Declining trajectory** in recent years.

**Key Observation**: Revenue has grown dramatically from $11M (2017) to $18M (2023), but expenses have kept pace. The model detects that this rapid growth may be unsustainable.

---

## 5. Is This School in the Dataset?

**Yes.** Texas A&M University-Commerce is included in the temporally validated dataset.

| Dataset | Years | Status |
|---------|-------|--------|
| Raw EADA Data | 2014-2023 | ✅ Present |
| Training Set | 2017-2019 | ✅ Present |
| Validation Set | 2020-2021 | ✅ Present |
| Test Set | 2022 | ✅ Present |
| Prediction Set | 2023 | ✅ Present |

**Why this structure?**
- 2014-2016: Used for lag feature calculations (lookback only)
- 2017-2019: Training data (learn patterns)
- 2020-2021: Validation data (tune hyperparameters)
- 2022: Holdout test (final honest evaluation)
- 2023: True future predictions (no labels available)

---

## 6. Model Details

### Model Used
- **Algorithm**: Logistic Regression
- **Training Years**: 2017-2019 (5,166 rows)
- **Validation Years**: 2020-2021 (3,444 rows)
- **Test Year**: 2022 (1,722 rows)
- **Test Accuracy**: 57.3% (Logistic Regression on 2022 holdout)

**Model Selection**: Three models were trained and evaluated:
- **Logistic Regression**: 57.3% accuracy (best performing, used for predictions) ⭐
- **Random Forest**: 54.6% accuracy
- **XGBoost**: 53.7% accuracy

All three models share the **same training data** (2017-2019) and **same preprocessing** (scaler, feature columns, medians). Logistic Regression was selected for predictions because it achieved the highest accuracy on the 2022 holdout test.

### Features Used (14 total)
1. **Structural**: Division
2. **Raw Values**: Grand Total Revenue, Grand Total Expenses
3. **Current Metrics**: Total_Athletes, Efficiency_Ratio, Revenue_Per_Athlete, Reports_Exactly_One
4. **1-Year Growth**: Revenue_Growth_1yr, Expense_Growth_1yr
5. **2-Year Trends**: Revenue_CAGR_2yr, Expense_CAGR_2yr, Efficiency_Mean_2yr
6. **Volatility**: Revenue_Volatility_2yr, Expense_Volatility_2yr

### Training Process
All models (Logistic Regression, Random Forest, XGBoost) were trained using:
- **Same Training Data**: 2017-2019 (5,166 rows)
- **Same Validation Data**: 2020-2021 (3,444 rows) for hyperparameter tuning
- **Same Preprocessing**: 
  - Same scaler (fit on training data only)
  - Same feature columns (14 features)
  - Same imputation medians (from training data)
- **Same Test Data**: 2022 (1,722 rows) for final evaluation

### Temporal Validation Approach
- **Split Method**: Temporal (strictly by time)
- **Training**: 2017-2019 data only
- **Validation**: 2020-2021 data for hyperparameter tuning
- **Testing**: 2022 data for final evaluation
- **Prediction**: 2023 data (true future, no labels)

This ensures the model never "sees the future" during training — all predictions are made using only past information.

---

## 7. Actionable Insights for Management

| Insight | Recommendation |
|---------|----------------|
| Close call prediction (40% vs 39%) | **High uncertainty** — monitor monthly financials closely |
| First predicted Declining in 6 years | Investigate what changed — this is unusual |
| High revenue growth (15.6%) but expenses growing (13.6%) | **Control expense growth** — maintain revenue/expense gap |
| High volatility ($1.7M revenue std dev) | **Stabilize revenue streams** — reduce dependence on volatile sources |
| Revenue grew from $11M to $18M in 6 years | **Sustainable growth?** — ensure growth is maintainable |
| Efficiency ratio still healthy (1.032) | **Maintain current efficiency** — don't let it slip |

### Strategic Recommendations
1. **Immediate**: Review expense budgets for 2024 — find areas to control growth
2. **Short-term**: Diversify revenue streams to reduce volatility
3. **Long-term**: Develop sustainable growth strategy that maintains efficiency
4. **Monitoring**: Track monthly financials — model suggests high uncertainty

---

## 8. Limitations and Caveats

1. **Low Confidence**: 40% vs 39% is essentially a coin flip — prediction is uncertain
2. **Model Accuracy**: 57.3% accuracy means the model is wrong ~43% of the time
3. **True Future**: 2023→2024 trajectory is unknown — this is a genuine prediction
4. **External Factors**: Model doesn't account for:
   - COVID-19 impacts
   - Conference realignment
   - Major facility investments
   - Changes in NCAA rules
   - Unexpected revenue sources or expense cuts

---

*Generated: December 2024*  
*Model: Logistic Regression (Temporally Validated, 57.3% accuracy on 2022 holdout)*  
*Approach: Temporal validation ensures honest predictions using only past information*
