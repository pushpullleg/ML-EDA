# Temporal Validation Plan: True Predictive Accuracy Assessment

## Executive Summary

The current model shows **92.8% accuracy (1,598/1,722 correct)**, but this is **circular validation** - the model predicts labels that are calculated from the same input data. This plan outlines how to achieve **true temporal validation** using 2023 data that exists in the raw dataset.

---

## The Problem: Circular Validation

### Current Label Formula (from `01_Feature_Engineering_Basic.ipynb`)
```python
# Labels are derived FROM the input features
if (rev_growth > 0.03) and (exp_growth < rev_growth): 
    return 'Improving'
elif (rev_growth < 0.00) or (exp_growth > rev_growth + 0.03): 
    return 'Declining'
else: 
    return 'Stable'
```

### Why 92.8% is Misleading
- **Input:** 2022 revenue growth, expense growth, efficiency metrics
- **Output:** Label derived from those SAME 2022 metrics
- **Result:** Model learns the label formula, not future prediction

### What We Need
- **Input:** 2022 features (revenue, expenses, trends)
- **Output:** What ACTUALLY happened in 2023 (true future outcome)

---

## Data Discovery: 2023 Data EXISTS!

### Raw Data File
**Location:** `/archive/initial files/Output_10yrs_reported_schools_17220.csv`

### Year Distribution (from terminal command)
```
1722 2014
1722 2015
1722 2016
1722 2017
1722 2018
1722 2019
1722 2020
1722 2021
1722 2022
1722 2023  ← THIS IS WHAT WE NEED!
```

### Current Processed Data
**Location:** `/final/assets/data/trajectory_excellent.csv`
- Only contains 2017-2022 (2023 was dropped during processing)
- 10,332 rows = 1,722 schools × 6 years

---

## The Solution: 5-Step Temporal Validation

### Step 1: Reprocess Raw Data to Include 2023
```python
# Read the full raw dataset
raw_df = pd.read_csv('/archive/initial files/Output_10yrs_reported_schools_17220.csv')

# Keep 2017-2023 (need 2023 for true labels)
df = raw_df[raw_df['Survey_Year'] >= 2017].copy()
```

### Step 2: Create TRUE Future Labels
```python
# For each school-year, the label should be what ACTUALLY happened NEXT year
# Example: 2022 row gets labeled based on 2023 outcomes

def create_true_future_label(group):
    group = group.sort_values('Survey_Year')
    # Shift revenue/expense data to get NEXT year's actual values
    group['Next_Year_Revenue'] = group['Total_Revenue'].shift(-1)
    group['Next_Year_Expenses'] = group['Total_Expenses'].shift(-1)
    # Calculate what actually happened
    group['Actual_Future_Growth'] = (group['Next_Year_Revenue'] - group['Total_Revenue']) / group['Total_Revenue']
    return group

df = df.groupby('UNITID').apply(create_true_future_label)
```

### Step 3: Temporal Train/Test Split
```python
# Training: 2017-2021 data predicting 2018-2022 outcomes
train_df = df[df['Survey_Year'] <= 2021]

# Testing: 2022 data predicting 2023 outcomes (TRUE HOLDOUT)
test_df = df[df['Survey_Year'] == 2022]
```

### Step 4: Retrain Model
```python
# Use same feature engineering pipeline
# But now labels represent ACTUAL future outcomes

X_train = train_df[feature_columns]
y_train = train_df['True_Future_Label']  # What actually happened in next year

X_test = test_df[feature_columns]
y_test = test_df['True_Future_Label']  # What actually happened in 2023
```

### Step 5: Calculate TRUE Predictive Accuracy
```python
# This accuracy means: "Given 2022 data, how well do we predict 2023?"
model.fit(X_train, y_train)
predictions = model.predict(X_test)
true_accuracy = accuracy_score(y_test, predictions)

print(f"TRUE Predictive Accuracy: {true_accuracy:.1%}")
print(f"Correct Predictions: {sum(predictions == y_test)}/1,722")
```

---

## Expected Outcomes

### Realistic Accuracy Expectations
| Validation Type | Expected Accuracy | Meaning |
|-----------------|-------------------|---------|
| Current (circular) | 92.8% | Model learned the formula |
| Temporal (true) | 50-70% | Actual predictive power |
| Random baseline | 33% | Guessing among 3 classes |

### Why Lower is Actually Better
- **Honest assessment** of model capability
- **Identifies** which schools are truly predictable vs. volatile
- **Useful** for real-world decision making

---

## Files to Modify/Create

### New Notebook: `07_Temporal_Validation.ipynb`
1. Load raw data with 2023
2. Create true future labels
3. Implement temporal split
4. Retrain and evaluate
5. Compare with current accuracy

### Update: `06_Full_Dataset_Visualization.ipynb`
- Add comparison: circular vs. temporal accuracy
- Visualize which predictions were truly correct

---

## Key Questions This Will Answer

1. **Can we actually predict the future?** (vs. just learning a formula)
2. **Which schools are predictable?** (consistent trajectories)
3. **Which features matter for TRUE prediction?** (not just formula reconstruction)
4. **What's our real-world utility?** (for athletic directors, policy makers)

---

## Action Items for Next Session

- [ ] Create `07_Temporal_Validation.ipynb`
- [ ] Reprocess raw data to include 2023
- [ ] Implement true future labels
- [ ] Create temporal train/test split
- [ ] Retrain model
- [ ] Calculate and report TRUE accuracy
- [ ] Update final report with honest assessment

---

*Created: Session identifying circular validation issue*
*Purpose: Enable true predictive accuracy assessment using 2023 data*
