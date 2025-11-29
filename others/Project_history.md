Project Narrative (Clear, Chronological, and Fully Integrated)
1. How the project started

We began with the intention of doing an ML project that fits within the course constraints:
• around 10,000 rows
• around 10 features
• something manageable, ideally in the ML/time-series domain.

Our early conversations revolved around time-series ideas, but nothing was fixed.

2. Discovery of the EADA dataset

I found the EADA dataset and brought it to our discussions.
Our immediate questions were:
• Does this dataset meet the row/feature constraints?
• Can it support a meaningful ML problem?
• Is it clean enough to build on?

I took responsibility for exploring, cleaning, and preparing the dataset so we could use it in our proposal.

At that stage, without deeper analysis, we chose the placeholder idea:
Predict institutional “efficiency.”

This was simply to have a concrete direction for the proposal.

3. Project proposal submission

We submitted the proposal with:
• The cleaned dataset
• The temporary “efficiency prediction” idea
• No deep justification for technique choice yet
(because we hadn’t done EDA or explored the data properly)

This lack of justification became important during the next step.

4. First meeting with the professor

During our meeting, the professor asked:
• Why did you choose this technique?
• What evidence do you have that this method fits the data?
• Where is your initial EDA or MLEDA?

I explained that the technique surfaced from online research.
She clarified that technique selection must follow EDA, not the other way around.
She also highlighted the difference between classroom ML topics and current real-world ML techniques.

This meeting made it clear that our problem statement needed to be data-driven, not assumption-driven.

5. Post-meeting: EDA and natural evolution of the problem

I went back and conducted structured EDA.
Based on the patterns in the data, one insight became obvious:
"Efficiency" is not a strong or predictable target for this dataset.

Instead, the dataset showed:
• consistent year-to-year financial patterns
• clear trajectories (improving, stable, declining)

That naturally led to a much more meaningful ML framing.

Finalized Problem Statement (After EDA)

Predict whether a college athletic department’s financial performance will get better, get worse, or stay the same next year — so administrators can act early.

This new problem statement is:
• grounded in the data
• aligned with real-world use
• measurable
• appropriate for ML classification
• more insightful than “efficiency prediction”

This is the problem we agreed to carry forward.

8. Second meeting with the professor

This second meeting did not change our direction.
My teammate presented:
• EDA summary
• features
• correlations

The professor commented generally on:
• positive vs. negative correlations
• how both matter
• high-level modeling perspective

She did not discuss or question our new problem statement.
There was no deep guidance or critique.
The discussion stayed at a surface level, and I mostly observed without participating.

So our classification problem remains intact, and no contradictions were raised.

6. Sharing code, assumptions, and roles

While exploring further, I experimented with models such as:
• Linear/logistic classification
• Random Forest
• XGBoost

Most of this was for my own learning and understanding.
I used LLMs to speed up writing code, testing ideas, and debugging.

My assumption was:
• I handle dataset discovery, cleaning, EDA, and problem framing
• You lead the modeling logic and technique selection, since you completed ML coursework

I shared my exploratory code with you so we could stay aligned.
If something was useful, good.
If not, it was optional.

7. Miscommunications

At some point, you said, “Don’t use my code; don’t follow it.”
I said the same — there is no expectation for either of us to follow each other’s exact code.

The misunderstandings came mainly from:
• different working styles
• asynchronous communication
• missing context
• code being shared without clear purpose

Clarifying this helps reduce friction going forward.

9. Current technical status

Where the project stands today:

Confirmed problem:
Classification of institutions into Improving / Stable / Declining financial trajectory for the next year.

Models implemented and evaluated:
• Logistic Regression + SMOTE (baseline ML) → 54.6% accuracy, 0.653 ROC-AUC
• Random Forest + SMOTE → 88.0% accuracy, 0.968 ROC-AUC
• XGBoost + SMOTE (final model) → 87.6% accuracy, 0.968 ROC-AUC, 0.827 Macro F1

All models exceed A-grade requirements (>70% accuracy, >0.75 ROC-AUC, >0.70 Macro F1, >0.50 Improving F1).

Key techniques used:
• SMOTE for class imbalance (Improving class was underrepresented)
• Temporal lag features (Year t-1, t-2 data to predict Year t trajectory)
• Stratified train/test split (80/20)
• SHAP for model interpretability
• One-hot encoding for categorical features (Division)

Deliverables completed:
• Final dataset: 10,332 rows × 52 features
• Trained model: trajectory_model.joblib (XGBoost + SMOTE pipeline)
• CLI deployment script with regression test
• Full documentation (Report, Appendices A-E)
• Figures: Confusion matrix, model comparison, feature importance, SHAP summary

This is the shared, transparent understanding of where the project is and how it evolved.