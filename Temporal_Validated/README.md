# Temporal_Validated: NCAA Athletics Financial Trajectory Prediction

## Overview

This project predicts the **financial trajectory** (Improving, Stable, Declining) of NCAA athletic programs using temporal validation to ensure honest, leak-free predictions.

## Why This Project Exists

The original `final/` project achieved 87-92% accuracy but had **critical data leakage**:
- Used `Lag1_Target_Label` as a feature (the answer encoded as input!)
- Used random `train_test_split()` instead of temporal split (time travel)

This project rebuilds from scratch with proper temporal validation.

## Project Structure

```
Temporal_Validated/
├── PLAN.md                    # What went wrong and how we fixed it
├── README.md                  # This file
├── data/
│   ├── raw/                   # Original EADA data
│   │   └── Output_10yrs_reported_schools_17220.csv
│   └── processed/             # Cleaned and split data
│       ├── features_temporal.csv
│       ├── features_2023_predict.csv
│       ├── train.csv          # 2017-2019
│       ├── val.csv            # 2020-2021
│       └── test.csv           # 2022
├── notebooks/
│   ├── 00_Data_Loading_and_Overview.ipynb
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Feature_Discovery.ipynb
│   ├── 03_Target_Definition.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Temporal_Split.ipynb
│   ├── 06_Model_Training.ipynb
│   ├── 07_Model_Evaluation.ipynb
│   ├── 08_Predictions.ipynb
│   └── leakage_check.ipynb    # Verification notebook
├── models/                    # Trained models
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── reports/                   # Results and visualizations
│   ├── evaluation_results.csv
│   ├── predictions_2023.csv
│   └── *.png                  # Figures
└── Report.md                  # Comprehensive findings
```

## Quick Start

1. **Read the Plan**: Start with `PLAN.md` to understand what went wrong
2. **Run Notebooks**: Execute notebooks 00-08 in order
3. **Verify**: Run `leakage_check.ipynb` to confirm no data leakage
4. **Review Results**: See `Report.md` for findings

## Year Allocation

| Years | Purpose | Rows (~) |
|-------|---------|----------|
| 2014-2016 | Lag lookback (consumed) | N/A |
| 2017-2019 | Training | 5,166 |
| 2020-2021 | Validation | 3,444 |
| 2022 | Holdout Test | 1,722 |
| 2023 | Future Prediction | 1,722 |

## Key Differences from `final/`

| Aspect | `final/` (Flawed) | `Temporal_Validated` (Correct) |
|--------|-------------------|-------------------------------|
| Accuracy | 87-92% (fake) | 55-65% (real) |
| Split Method | Random | Temporal |
| Feature `Lag1_Target_Label` | ✅ Used | ❌ Forbidden |
| 2023 Predictions | Dropped | Generated |
| Validation | None | 5 verification checks |

## Features Used (15 total)

### Safe Features (No Leakage)
- **Structural**: Division
- **Current Metrics**: Efficiency_Ratio, Revenue_Per_Athlete, Total_Athletes
- **Growth (1yr)**: Revenue_Growth_1yr, Expense_Growth_1yr
- **Trends (2yr)**: Revenue_CAGR_2yr, Expense_CAGR_2yr, Efficiency_Mean_2yr
- **Volatility**: Revenue_Volatility_2yr, Expense_Volatility_2yr

### Forbidden Features (Cause Leakage)
- ❌ Lag1_Target_Label
- ❌ Same_Trajectory_As_Lag
- ❌ Trajectory_Changed
- ❌ Any feature containing "Target" or "Future"

## Results

### Expected vs Actual Accuracy
- **Random Baseline**: 33.3%
- **Our Models**: 55-65%
- **Improvement**: +22-32% over random

### Why Lower Accuracy is Good
The lower accuracy proves the model is working correctly:
- Financial trajectory is inherently hard to predict
- 55-65% accuracy represents genuine predictive value
- The 87-92% in `final/` was an artifact of data leakage

## Running the Analysis

```bash
# Navigate to project
cd "ML EDA/Temporal_Validated"

# Run notebooks in order
# Use Jupyter Notebook or VS Code with Jupyter extension
```

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn (for SMOTE)
joblib
```

## Authors

This project was rebuilt to fix data leakage issues discovered in the original `final/` analysis.

## License

Educational use only.
