# CSCI 538 Final Project

# Predicting NCAA Athletic Department Financial Trajectories: A Machine Learning Approach to Identify Improving, Stable, and Declining Programs

## Group Members: Mukesh Ravichandran, Abner Lusung

## East Texas A&M University

## December 2025

---

**Formatting Note for Word Conversion:**
- Standard size: 8.5 x 11"
- Page Margins: 1" on all sides
- Font: 12 pt., Times New Roman
- Spacing: 1.15
- Use APA-style headings throughout

---

# Abstract

This project develops a machine learning system to predict whether an NCAA athletic department is on an Improving, Stable, or Declining two-year financial trajectory, enabling administrators to act before budgets spiral. Using 10 years of NCAA EADA data (17,220 records from 1,722 institutions), we implemented strict temporal validation by training on 2017-2019 data, validating on 2020-2021, and testing on 2022 holdout data. Our best model (Logistic Regression) achieved 57.3% accuracy, representing a 29% improvement over random baseline. We engineered 14 leak-free features and verified our methodology through seven independent checks to ensure no data leakage. Finally, we generated predictions for all 1,722 institutions for 2023, with 45 high-confidence predictions identified for actionable decision-making. Our work demonstrates that proper temporal validation is essential for trustworthy machine learning in financial forecasting applications.

---

# 1. Introduction and Background

## 1.1 The Problem We Tried to Solve

Predicting the financial trajectory of NCAA athletic programs is crucial for university administrators, athletic directors, and policy makers who must allocate resources and plan for institutional sustainability. With over 1,700 NCAA member institutions managing billions of dollars in athletic revenues and expenses, the ability to forecast whether a program will improve, remain stable, or decline financially has significant practical implications.

A critical challenge in building such predictive models is avoiding **data leakage**—a subtle but devastating methodological flaw where information from the target variable or future time periods inadvertently influences the training features. Data leakage is particularly problematic in time series prediction, where two common mistakes can inflate accuracy:

1. **Target-Derived Features**: Using features derived from the target variable (such as lagged target labels) essentially provides the model with encoded answers as input.

2. **Random Train-Test Split**: Using random splitting instead of temporal splitting allows future data into training, enabling the model to "predict" the past using future knowledge.

Our research objective was to build a financial trajectory prediction system using proper temporal validation methodology, ensuring that our predictions are honest, reliable, and genuinely actionable for decision-makers.

## 1.2 Results from the Literature

The problem of data leakage in machine learning has been extensively documented in the research literature. Kaufman et al. (2012) identified leakage as one of the most common pitfalls in applied machine learning, noting that it often produces models that appear to perform exceptionally well during development but fail catastrophically in production. This motivated our careful approach to feature engineering and data splitting in this project.

Temporal validation is essential for time series prediction problems. Tashman (2000) established that rolling-origin evaluation—where models are trained only on past data and evaluated on future data—provides the most realistic assessment of forecasting accuracy. Bergmeir and Benítez (2012) further demonstrated that cross-validation techniques appropriate for i.i.d. data can produce severely biased estimates when applied to time series, supporting our decision to implement strict temporal splitting.

In the domain of financial prediction, Gu et al. (2020) showed that machine learning methods can provide meaningful predictive power for financial outcomes, though they emphasized that realistic out-of-sample testing is crucial for valid conclusions. Their work on asset return prediction achieved modest but statistically significant improvements over baseline models, consistent with our findings that genuine predictive accuracy in financial domains is typically modest (55-65% for multi-class problems) but still provides meaningful value over random guessing.

For addressing class imbalance in our three-class prediction problem, we employed SMOTE (Synthetic Minority Over-sampling Technique), introduced by Chawla et al. (2002). This technique generates synthetic examples of minority classes to balance training data without simply duplicating existing examples.

Our ensemble methods draw from foundational work by Breiman (2001) on Random Forests and Chen and Guestrin (2016) on XGBoost, both of which have demonstrated strong performance across diverse prediction tasks while offering interpretable feature importance measures.

## 1.3 Tools and Programs Available

Our implementation leverages the Python data science ecosystem:

- **Data Processing**: pandas (Wes McKinney, 2010) for data manipulation and numpy for numerical operations
- **Machine Learning**: scikit-learn (Pedregosa et al., 2011) for model training, preprocessing, and evaluation
- **Gradient Boosting**: XGBoost (Chen & Guestrin, 2016) for advanced ensemble methods
- **Class Imbalance**: imbalanced-learn library for SMOTE implementation
- **Visualization**: matplotlib and seaborn for statistical graphics
- **Development Environment**: Jupyter notebooks for reproducible analysis

Our methodology follows best practices for time series machine learning:
1. Implementing strict temporal validation rather than random splitting
2. Excluding all target-derived features from the feature set
3. Preserving 2023 data for true future predictions (since labels require 2024 data)
4. Including comprehensive leakage verification checks throughout the pipeline

---

# 2. Overview of the Architecture

Our system architecture consists of a modular pipeline of nine Jupyter notebooks, each performing a specific function in the machine learning workflow. This design ensures reproducibility, clear documentation, and easy verification of each processing step.

## 2.1 Finished Work: Running Modules

The complete pipeline includes the following executable modules:

**Data Understanding Phase:**
- `00_Data_Loading_and_Overview.ipynb`: Loads raw NCAA EADA data (17,220 records, 580 columns) and documents data structure
- `01_Exploratory_Data_Analysis.ipynb`: Statistical analysis of distributions, correlations, and temporal patterns
- `02_Feature_Discovery.ipynb`: Identification of potential predictive features based on EDA observations

**Feature Engineering Phase:**
- `03_Target_Definition.ipynb`: Definition of three-class target (Improving, Stable, Declining) with threshold justification
- `04_Feature_Engineering.ipynb`: Creation of 14 leak-free features using only historical data

**Model Development Phase:**
- `05_Temporal_Split.ipynb`: Implementation of time-based data splitting
- `06_Model_Training.ipynb`: Training of three models with SMOTE balancing

**Evaluation and Prediction Phase:**
- `07_Model_Evaluation.ipynb`: Holdout evaluation on 2022 data (first and only use)
- `08_Predictions.ipynb`: Generation of 2023 predictions for all 1,722 institutions

**Verification:**
- `leakage_check.ipynb`: Seven independent verification checks confirming no data leakage

**Saved Artifacts:**
- `models/scaler.pkl`: StandardScaler fitted on training data
- `models/logistic_regression.pkl`, `random_forest.pkl`, `xgboost.pkl`: Trained models
- `models/feature_columns.pkl`, `train_medians.pkl`: Preprocessing artifacts

## 2.2 Work in Progress: Modules Designed but Not Implemented

- **Hyperparameter Optimization Module**: Grid search or Bayesian optimization for model tuning
- **Cross-Validation within Temporal Folds**: Time-series cross-validation for more robust validation estimates

## 2.3 Future Work: Modules a Future Continuation May Have

- **Ensemble Voting Classifier**: Combining predictions from multiple models
- **Deep Learning Module**: LSTM or Transformer architectures for sequence modeling
- **Real-time Prediction API**: Flask/FastAPI endpoint for live predictions
- **2024 Validation Module**: Comparing 2023 predictions against actual outcomes when data becomes available

---

# 3. Data Collection

## 3.1 Data Source

Our dataset comes from the **NCAA Equity in Athletics Disclosure Act (EADA)** database, which requires all co-educational postsecondary institutions that receive Title IV funding and have intercollegiate athletic programs to report annual financial and participation data.

**Citation**: U.S. Department of Education, Office of Postsecondary Education. (2024). Equity in Athletics Data Analysis. Retrieved from https://ope.ed.gov/athletics/

## 3.2 Dataset Characteristics

| Characteristic | Value |
|---------------|-------|
| File Name | `Output_10yrs_reported_schools_17220.csv` |
| Total Records | 17,220 |
| Unique Institutions | 1,722 |
| Time Span | 2014-2023 (10 years) |
| Original Features | 580 columns |
| Engineered Features | 14 columns |

## 3.3 Key Variables

**Financial Variables:**
- Grand Total Revenue: Total athletic revenue for the institution
- Grand Total Expenses: Total athletic expenses for the institution

**Structural Variables:**
- UNITID: Unique institution identifier
- Institution Name: Name of the institution
- State: State location
- Division: NCAA division (D1, D2, D3, Other)

**Participation Variables:**
- Total Athletes: Count of student athletes

## 3.4 Data Quality

The EADA data is self-reported by institutions and subject to federal reporting requirements, providing reasonable data quality. We handled missing values by:
1. Using median imputation with values calculated from training data only
2. Replacing infinite values (from division operations) with NaN before imputation
3. Excluding institutions with insufficient historical data for lag calculations

---

# 4. Methods and Implementation

## 4.1 Target Variable Definition

We predict the financial trajectory of each institution from year t to year t+1, classified into three categories:

| Class | Definition | Interpretation |
|-------|------------|----------------|
| **Improving** | Revenue growth > 3% AND Expense growth < Revenue growth | Financial position strengthening |
| **Declining** | Revenue growth < 0% OR Expense growth > Revenue growth + 3% | Financial position weakening |
| **Stable** | All other cases | Financial position maintaining |

These thresholds were selected to identify meaningful changes rather than noise from normal year-to-year variation.

## 4.2 Feature Engineering

We engineered 14 features that use only current and past data, explicitly avoiding any information leakage:

**Current Year Metrics (4 features):**
- `Efficiency_Ratio`: Revenue / Expenses
- `Revenue_Per_Athlete`: Revenue / Total Athletes
- `Total_Athletes`: Count of student athletes
- `Reports_Exactly_One`: Binary indicator for single-sport institutions

**One-Year Growth (2 features):**
- `Revenue_Growth_1yr`: (Revenue[t] - Revenue[t-1]) / Revenue[t-1]
- `Expense_Growth_1yr`: (Expenses[t] - Expenses[t-1]) / Expenses[t-1]

**Two-Year Trends (3 features):**
- `Revenue_CAGR_2yr`: Compound annual growth rate over 2 years
- `Expense_CAGR_2yr`: Compound annual growth rate over 2 years
- `Efficiency_Mean_2yr`: Average efficiency ratio over 2 years

**Volatility Measures (2 features):**
- `Revenue_Volatility_2yr`: Standard deviation of revenue over 2 years
- `Expense_Volatility_2yr`: Standard deviation of expenses over 2 years

**Raw Values (2 features):**
- `Grand Total Revenue`: Absolute revenue amount
- `Grand Total Expenses`: Absolute expense amount

**Categorical (1 feature):**
- `Division`: NCAA division encoded as ordinal (D1=3, D2=2, D3=1, Other=0)

**Forbidden Features** (explicitly excluded to prevent leakage):
- Any feature containing "Target" or "Label"
- `Lag1_Target_Label`, `Same_Trajectory_As_Lag`, `Trajectory_Changed`
- Any forward-looking features using shift(-1)

## 4.3 Temporal Split Strategy

We implemented strict temporal validation to ensure the model never sees future data during training:

| Split | Years | Rows | Purpose |
|-------|-------|------|---------|
| **Training** | 2017-2019 | 5,166 | Learn patterns |
| **Validation** | 2020-2021 | 3,444 | Tune hyperparameters |
| **Test (Holdout)** | 2022 | 1,722 | Final honest evaluation |
| **Future Prediction** | 2023 | 1,722 | True predictions (no labels) |

Years 2014-2016 are consumed for calculating 2-year lag features and do not appear in the final dataset.

## 4.4 Model Selection

We trained three classification models, each with different inductive biases:

**Logistic Regression:**
- Solver: L-BFGS
- Multi-class: Multinomial
- Class weights: Balanced
- Regularization: L2 (default)

**Random Forest:**
- Trees: 100
- Max depth: 10
- Class weights: Balanced
- Bootstrap: True

**XGBoost:**
- Estimators: 100
- Max depth: 5
- Learning rate: 0.1
- Objective: Multi-class softmax

## 4.5 Class Imbalance Handling

The target classes were imbalanced (approximately 28% Declining, 46% Stable, 26% Improving). We applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to create a balanced training set, while keeping validation and test sets in their original distributions for realistic evaluation.

## 4.6 Algorithm Selection Justification

We chose traditional machine learning methods over deep learning for several reasons:

1. **Dataset Size**: With ~5,000 training samples, traditional methods often outperform deep learning
2. **Interpretability**: Feature importance from RF and XGBoost aids understanding
3. **Computational Efficiency**: Faster training and inference
4. **Baseline Establishment**: Before complex models, establishing what simpler methods achieve

---

# 5. Results and Evaluation

## 5.1 Evaluation Metrics

We evaluated models using multiple metrics to capture different aspects of performance:

- **Accuracy**: Overall proportion of correct predictions
- **F1 Score (Weighted)**: Harmonic mean of precision and recall, weighted by class frequency
- **F1 Score (Macro)**: Unweighted average F1 across all classes
- **Precision and Recall**: Per-class and weighted averages

## 5.2 Model Performance on 2022 Holdout

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Precision | Recall |
|-------|----------|---------------|------------|-----------|--------|
| **Logistic Regression** | **57.3%** | **0.563** | 0.487 | 0.600 | 0.573 |
| Random Forest | 54.6% | 0.560 | **0.498** | 0.592 | 0.546 |
| XGBoost | 53.7% | 0.551 | 0.491 | 0.581 | 0.537 |

Logistic Regression achieved the highest accuracy (57.3%), while Random Forest had the best macro F1 (0.498), indicating more balanced performance across classes.

## 5.3 Baseline Comparison

| Approach | Accuracy | Improvement |
|----------|----------|-------------|
| Random Guessing | 33.3% | — |
| Most-Frequent Class (Stable) | 28.3% | — |
| **Our Best Model (LR)** | **57.3%** | **+29.0%** over baseline |

Our model provides a genuine 29 percentage point improvement over random guessing, demonstrating real predictive value despite the inherently difficult prediction problem.

## 5.4 Interpreting Model Accuracy

The 57.3% accuracy represents meaningful predictive power for this challenging problem:

1. **Financial trajectories are inherently unpredictable**: External factors (economic conditions, coaching changes, conference realignment) significantly impact outcomes
2. **Three-class problem is harder than binary**: Random baseline is 33.3%, not 50%
3. **No information leakage**: Our features contain no future information, ensuring honest evaluation
4. **Temporal validation is realistic**: Our evaluation matches how the model would be used in practice—predicting future outcomes from past data

## 5.5 Feature Importance

The top predictive features (from Random Forest importance) were:
1. Revenue_Growth_1yr (most predictive)
2. Expense_Growth_1yr
3. Grand Total Revenue
4. Efficiency_Ratio
5. Revenue_CAGR_2yr

This aligns with domain intuition: recent growth rates are the strongest predictors of future trajectory.

## 5.6 2023 Predictions

We generated predictions for all 1,722 institutions for 2023:

| Predicted Trajectory | Count | Percentage |
|---------------------|-------|------------|
| Stable | 803 | 46.6% |
| Declining | 473 | 27.5% |
| Improving | 446 | 25.9% |

**High-Confidence Predictions**: 45 institutions have prediction confidence ≥ 70%, making them most actionable for stakeholders.

**Top High-Confidence Declining Predictions:**
- Thomas University: 90.0% confidence
- University of Olivet: 89.0% confidence
- Miles College: 75.0% confidence

These predictions await validation when 2024 financial data becomes available.

## 5.7 Case Study: Predicting 2023 Financial Trajectories

Our model uses historical data from 2014–2022 to generate true future predictions for 2023:

| Data Role | Years | Records | Purpose |
|-----------|-------|---------|---------|
| Feature Lookback | 2014–2016 | — | Calculate 2-year lag features (no direct training rows) |
| Training | 2017–2019 | 5,166 | Learn patterns |
| Validation | 2020–2021 | 3,444 | Tune model and check generalization |
| Testing | 2022 | 1,722 | Final evaluation (57.3% accuracy) |
| **Prediction** | **2023** | **1,722** | **True future predictions (no labels yet)** |

Using patterns learned from 2017–2022, the model predicts financial trajectories for 2023. These predictions will be validated when 2024 financial data becomes available.

### 5.7.1 Example Institutions by Confidence Level

To illustrate how the model behaves across different confidence levels, we highlight several well-known institutions, including our own university.

**High-Confidence Predictions (≥ 60% confidence):**

| Institution | Division | 2023 Predicted Trajectory | Confidence Score |
|------------|----------|---------------------------|------------------|
| The University of Alabama | D1 | Declining | 67.5% |
| University of Georgia | D1 | Declining | 62.5% |
| Alabama A & M University | D1 | Stable | 61.3% |

In these cases, the model assigns relatively high probability (≥ 60%) to a single trajectory class, indicating stronger confidence in the prediction.

**Moderate-Confidence Predictions (≈ 50–60% confidence):**

| Institution | Division | 2023 Predicted Trajectory | Confidence Score |
|------------|----------|---------------------------|------------------|
| Stanford University | D1 | Declining | 54.1% |
| Duke University | D1 | Declining | 54.6% |
| Michigan State University | D1 | Declining | 53.6% |

Here the model leans toward a single class, but the confidence is only slightly above 50%. These predictions are informative but should be interpreted with caution.

**Uncertain Predictions (< 50% confidence):**

| Institution | Division | 2023 Predicted Trajectory | Confidence Score |
|------------|----------|---------------------------|------------------|
| Texas A & M University-Commerce | D1 | Declining | 39.9% |
| Ohio State University | D1 | Improving | 46.9% |
| University of Notre Dame | D1 | Improving | 43.3% |

For these institutions, the model's confidence is below 50%, meaning the predicted class is only slightly more likely than the alternatives. In practice, such predictions should be treated as **highly uncertain** and used more as prompts for further investigation than as firm conclusions.

#### Confidence Threshold Rationale

We established confidence thresholds based on the three-class prediction problem:

- **Random baseline**: In a balanced three-class problem, random guessing assigns ~33.3% probability to each class
- **≥60% (High Confidence)**: More than 1.8× the random baseline, indicating the model is relatively certain
- **50-60% (Moderate Confidence)**: Above random but below strong certainty; the model leans toward one class but with notable uncertainty
- **<50% (Uncertain)**: Below the threshold where one class is clearly favored; predictions are only slightly better than random

These thresholds align with standard practice in multi-class classification where confidence scores represent the maximum class probability. The ≥70% threshold for "high-confidence" predictions (used in Section 5.6) represents an even more conservative standard, identifying the 45 institutions where the model is most certain.

Overall, this case study demonstrates:
- How several years of historical data (2014–2022) are used to make a single-year forecast (2023).
- That the model can produce both high-confidence and low-confidence predictions, and that confidence scores are critical for responsible interpretation.
- How stakeholders (such as administrators at Texas A & M University-Commerce) can use both the predicted trajectory **and** the confidence level to decide how much weight to place on a given prediction.

---

# 6. Achievements and Observations

## 6.1 Key Technical Achievements

1. **Complete Temporal Validation Pipeline**: Implemented proper time-based splitting that prevents all forms of temporal leakage
2. **Seven Verification Checks**: All leakage checks passed:
   - No target-derived features
   - No future-looking features
   - Temporal split integrity verified
   - No year overlap between splits
   - Year not used as feature
   - Accuracy in realistic range (45-75%)
   - Model beats baseline

3. **True Future Predictions**: Generated actionable predictions for 2023 using proper methodology

4. **Reproducible Pipeline**: Nine documented notebooks with clear progression

## 6.2 Lessons Learned

1. **Data leakage is subtle and devastating**: Without careful methodology, models can appear to perform exceptionally well while providing no real predictive value
2. **Honest accuracy provides real value**: 57.3% accuracy that stakeholders can trust enables genuine decision support
3. **Temporal validation is non-negotiable**: For any time series prediction, temporal splitting is essential to obtain realistic performance estimates

## 6.3 Individual Contributions

**Mukesh Ravichandran:**
- Led data exploration and EDA (notebooks 00-02)
- Implemented feature engineering pipeline (notebook 04)
- Conducted model evaluation and generated final predictions (notebooks 07-08)
- Designed and executed leakage verification checks
- Primary author of project report

**Abner Lusung:**
- Defined target variable and classification thresholds (notebook 03)
- Implemented temporal split methodology (notebook 05)
- Led model training and hyperparameter selection (notebook 06)
- Created visualizations for results presentation
- Prepared presentation materials

---

# 7. Discussion and Conclusions

## 7.1 Summary of Findings

This project successfully developed an NCAA financial trajectory prediction system with proper temporal validation. Our key findings are:

1. **Proper methodology prevents data leakage**: By using temporal splitting and excluding target-derived features, we ensured honest evaluation
2. **Achieved 57.3% accuracy**: This represents genuine predictive power, improving 29% over random baseline (33.3%)
3. **Temporal validation is essential**: Strict time-based splitting ensures realistic performance estimates that match real-world deployment
4. **Predictions are actionable**: 1,722 institutions have predictions for 2023, with 45 high-confidence cases ready for stakeholder attention

## 7.2 Limitations

1. **External Factors Not Modeled**: Economic conditions, policy changes, and institutional decisions significantly impact financial trajectories but are not captured in our features
2. **COVID-19 Impact**: The 2020-2021 validation period was affected by pandemic disruptions, potentially affecting model calibration
3. **Self-Reported Data**: EADA data quality depends on institutional reporting practices
4. **57.3% Accuracy Uncertainty**: While better than baseline, significant prediction uncertainty remains

## 7.3 Future Extensions

1. **Feature Engineering**: Incorporate macroeconomic indicators, conference-level features, and institutional characteristics
2. **Advanced Models**: Explore LSTM networks for sequence modeling or attention mechanisms
3. **Ensemble Methods**: Combine predictions from multiple models using voting or stacking
4. **Validation Study**: Compare 2023 predictions against 2024 actual outcomes when available
5. **Real-time System**: Develop an API for continuous prediction updates

## 7.4 Practical Implications

For NCAA athletic administrators, our model provides:
- Early warning indicators for financial decline risk
- Identification of institutions with strong improvement trajectories
- Decision support tool (not replacement) for resource allocation
- Framework for monitoring financial health trends

The 57.3% accuracy means predictions should inform but not dictate decisions, with high-confidence cases (≥70%) warranting closer attention.

---

# 8. References

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213. https://doi.org/10.1016/j.ins.2011.12.028

> This paper examines cross-validation methods for time series data, demonstrating that standard k-fold cross-validation produces biased estimates for temporal data. We applied their recommendations by using strictly temporal splits. Referenced in Section 1.2.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

> The foundational paper introducing Random Forest algorithm. We used Random Forest as one of our three models due to its robustness and interpretable feature importance. Referenced in Section 4.4.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. https://doi.org/10.1613/jair.953

> Original paper introducing SMOTE for handling class imbalance. We applied SMOTE to balance our training data across three classes. Referenced in Section 4.5.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

> Introduced the XGBoost algorithm, which we used as one of our ensemble methods. Referenced in Sections 1.2 and 4.4.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhaa009

> Comprehensive study of machine learning methods for financial prediction, demonstrating realistic accuracy expectations for financial forecasting. Our results align with their findings that honest predictive accuracy in financial domains is modest but meaningful. Referenced in Section 1.2.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21. https://doi.org/10.1145/2382577.2382579

> Formal treatment of data leakage in machine learning. This paper informed our methodology for avoiding target-derived features and ensuring proper data splitting. Referenced in Section 1.2.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

> Documentation for scikit-learn library used for model training and evaluation. Referenced in Section 1.3.

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437-450. https://doi.org/10.1016/S0169-2070(00)00065-0

> Established best practices for out-of-sample testing in forecasting, supporting our temporal validation approach. Referenced in Section 1.2.

U.S. Department of Education, Office of Postsecondary Education. (2024). Equity in Athletics Data Analysis. Retrieved from https://ope.ed.gov/athletics/

> Official source for NCAA EADA financial data used in this project. Referenced in Section 3.1.

---

# Appendices

## Appendix A: Individual Stories

### A.1 Mukesh Ravichandran

My primary contribution to this project was leading the data exploration and analysis pipeline, from initial data loading through final predictions. I focused extensively on understanding the challenges of data leakage in time series machine learning and ensuring our methodology avoided these common pitfalls.

I implemented the feature engineering notebook (04_Feature_Engineering.ipynb), carefully designing features that only use current and past data. This required thinking critically about what information would be available at prediction time versus what would constitute leakage. For example, I ensured that growth rates used only historical values and that no target-derived features were included.

The model evaluation and prediction phases (notebooks 07-08) were also my responsibility. I designed the evaluation framework to provide comprehensive metrics (accuracy, F1 scores, precision, recall) and created the baseline comparisons that demonstrate our model's genuine predictive value. Generating the 2023 predictions was particularly satisfying because these represent true forward-looking predictions that can inform real decisions.

The leakage verification notebook was perhaps my most important contribution. I designed seven independent checks that verify the integrity of our methodology, providing confidence that our 57.3% accuracy represents genuine predictive power. Through this project, I learned that methodological rigor is more valuable than impressive-looking results, and that data leakage can be incredibly subtle yet completely invalidate an analysis.

### A.2 Abner Lusung

My focus in this project was on the methodological foundations: defining what we're predicting, how we split the data, and how we train our models. The target definition (notebook 03) required careful thought about what constitutes meaningful financial trajectory change versus normal year-to-year variation. I established the 3% thresholds for Improving and Declining classes based on analysis of the historical data distribution.

Implementing the temporal split (notebook 05) was critical to ensuring proper methodology. I designed the year allocation strategy: 2014-2016 for lag calculation, 2017-2019 for training, 2020-2021 for validation, 2022 for testing, and 2023 for true predictions. This ensures strict temporal separation and prevents any form of future information from leaking into training.

For model training (notebook 06), I implemented three algorithms with appropriate configurations and integrated SMOTE for handling class imbalance. I experimented with different hyperparameters and documented why we chose the final settings. The decision to use balanced class weights in addition to SMOTE helped improve performance on minority classes.

Creating the visualizations for our results was also my responsibility. The confusion matrices, model comparison charts, and feature importance plots help communicate our findings clearly. Through this project, I gained deep appreciation for the importance of proper experimental design in machine learning—our 57.3% accuracy represents genuine predictive value that stakeholders can rely on for decision-making.

---

## Appendix B: Data Dictionary

### B.1 Feature Definitions

| Feature | Type | Formula | Description |
|---------|------|---------|-------------|
| Division | Categorical | Encoded: D1=3, D2=2, D3=1, Other=0 | NCAA division classification |
| Grand Total Revenue | Numeric | Raw value | Total athletic revenue ($) |
| Grand Total Expenses | Numeric | Raw value | Total athletic expenses ($) |
| Total_Athletes | Numeric | Raw value | Count of student athletes |
| Efficiency_Ratio | Numeric | Revenue / Expenses | Financial efficiency measure |
| Revenue_Per_Athlete | Numeric | Revenue / Athletes | Revenue intensity per athlete |
| Reports_Exactly_One | Binary | 1 if single sport, 0 otherwise | Institution scope indicator |
| Revenue_Growth_1yr | Numeric | (Rev[t] - Rev[t-1]) / Rev[t-1] | One-year revenue change |
| Expense_Growth_1yr | Numeric | (Exp[t] - Exp[t-1]) / Exp[t-1] | One-year expense change |
| Revenue_CAGR_2yr | Numeric | (Rev[t] / Rev[t-2])^0.5 - 1 | Two-year compound growth |
| Expense_CAGR_2yr | Numeric | (Exp[t] / Exp[t-2])^0.5 - 1 | Two-year compound growth |
| Efficiency_Mean_2yr | Numeric | mean(Eff[t], Eff[t-1]) | Rolling efficiency average |
| Revenue_Volatility_2yr | Numeric | std(Rev[t], Rev[t-1]) | Revenue stability measure |
| Expense_Volatility_2yr | Numeric | std(Exp[t], Exp[t-1]) | Expense stability measure |

### B.2 Target Variable Definition

| Class | Label | Definition |
|-------|-------|------------|
| Declining | 0 | Revenue growth < 0% OR Expense growth > Revenue growth + 3% |
| Stable | 1 | Neither Improving nor Declining |
| Improving | 2 | Revenue growth > 3% AND Expense growth < Revenue growth |

---

## Appendix C: Verification Checklist

### C.1 Leakage Verification Results

| Check # | Verification | Status | Details |
|---------|--------------|--------|---------|
| 1 | No target-derived features | PASSED | 20 columns checked against 6 forbidden patterns |
| 2 | No future-looking features | PASSED | No Future_ or Next_ prefixes found |
| 3 | Temporal split integrity | PASSED | Train (2017-2019) < Val (2020-2021) < Test (2022) |
| 4 | No year overlap | PASSED | All 5 overlap checks passed |
| 5 | Year not used as feature | PASSED | 14 clean feature columns |
| 6 | Accuracy in realistic range | PASSED | 57.3% within expected 45-75% range |
| 7 | Model beats baseline | PASSED | +29.0% improvement over 33.3% baseline |

### C.2 Forbidden Feature Patterns

The following patterns were explicitly excluded from features:
- `Target` (any feature containing this substring)
- `Label` (any feature containing this substring)
- `Lag1_Target` (the specific leaky feature from original)
- `Same_Trajectory` (derived from target comparison)
- `Trajectory_Changed` (derived from target comparison)
- `Future_` (any forward-looking feature)

---

## Appendix D: File Structure

```
Temporal_Validated/
├── data/
│   ├── raw/
│   │   └── Output_10yrs_reported_schools_17220.csv
│   └── processed/
│       ├── features_temporal.csv
│       ├── features_2023_predict.csv
│       ├── train.csv (5,166 rows)
│       ├── val.csv (3,444 rows)
│       └── test.csv (1,722 rows)
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
│   └── leakage_check.ipynb
├── models/
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── train_medians.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── reports/
│   ├── evaluation_results.csv
│   ├── predictions_2023.csv
│   ├── predictions_2023_high_confidence.csv
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── temporal_split.png
└── docs/
    ├── CSCI538_Final_Report.md
    └── [Project templates and rubrics]
```

---

*Report prepared by Mukesh Ravichandran and Abner Lusung*
*East Texas A&M University - CSCI 538 Final Project*
*December 2025*
