# Financial-Health-Prediction
A multiclass classification model predicting SME financial resilience (Low/Medium/High) across four Southern African economies, using survey data on savings, debt, credit access, and shock exposure. Built with LightGBM; includes full EDA and domain-driven feature engineering.
## Approach

**1. Data Understanding**
Started by validating structural integrity, confirmed the target (FHI) exists only 
in train, checked shapes, dtypes, and memory footprint of both sets before touching 
the data.

**2. Target Variable Analysis**
Examined the FHI class distribution to check for imbalance, since this determines 
whether stratified sampling and class-weighting are needed downstream.

**3. Missing Data & Feature Profiling**
Quantified missingness per column, then separated numerical vs. categorical features 
to profile each separately, distributions and summary stats for numerical, 
cardinality for categorical.

**4. Cross-Country Patterns**
Since the data spans four countries (Eswatini, Lesotho, Malawi, Zimbabwe), checked 
whether class distribution and feature coverage differ by country. This is relevant 
given that the survey instrument may not be perfectly uniform across economies.

**5. Feature vs. Target Relationships**
Grouped key numerical and categorical features by FHI class to surface early signal 
on which variables might be predictive.

**6. Data Quality Checks**
Verified no duplicate or overlapping IDs between train/test to rule out leakage.

**7. Feature Engineering**
Built domain-informed features (such as the profit margin ratio) rather than relying 
purely on raw survey responses, since financial health is inherently a ratio/derived 
concept, not a single-variable outcome.

**8. Modeling**
Trained a LightGBM classifier, chosen for native categorical feature support (no 
manual encoding needed) and strong baseline performance on tabular survey data. 
Split was stratified by both country and target to preserve class balance across 
regions.

**9. Evaluation**

The model achieved **88% overall accuracy** on the validation set (1,924 samples), 
with strong but uneven performance across classes:

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| High   | 0.90      | 0.59   | 0.71     | 93      |
| Low    | 0.89      | 0.97   | 0.92     | 1,256   |
| Medium | 0.84      | 0.72   | 0.78     | 575     |

**High** was the hardest class to predict, despite strong precision (0.90, meaning 
predictions of "High" were usually correct), recall was only 0.59, so the model missed 
over 40% of actual High-FHI businesses. The confusion matrix shows most of these were 
misclassified as Medium (36 of 93 High cases), suggesting the boundary between High 
and Medium financial health is the model's main weak point, likely because High is 
also the smallest class (only 93 of 1,924 samples), giving the model less signal to 
learn from.

**Low** was predicted most reliably (F1 = 0.92), consistent with it being the majority 
class (1,256 samples) with the clearest feature separation.

**Top predictive features** were dominated by direct financial indicators rather than 
demographics: `business_turnover`, `business_expenses`, `owner_age`, `personal_income`, 
and the engineered `profit_margin` and `financial_access_score` features ranked highest, 
validating that the custom feature engineering (profit margin, financial access score) 
added real signal. Insurance-related and attitude/perception variables contributed 
moderately; `country` was the least informative feature.

## Key takeaways
- Class imbalance (High = 93 vs. Low = 1,256) is the main driver of the High-class 
  recall gap.
- Engineered features (profit margin, financial access score) rank among the top 6 
  predictors, ahead of most raw survey responses. This confirms the value of domain-driven 
  feature engineering over relying on raw inputs alone.
- The model confuses High↔Medium far more than Low↔High, suggesting these two classes 
  sit close together in feature space.


# Data Access

This project uses data from the data.org Financial Health Prediction Challenge on Zindi.
The data is sourced from SME surveys and includes detailed information about business owners, 
their financial habits, exposure to risks, access to credit, and overall business performance.
Data is not included in this repository (Zindi's standard Terms of Use state that competition 
data is the sole property of Zindi and the competition host, and you may not transmit, 
duplicate, publish, redistribute, or make it available to anyone not participating in the 
competition) download it directly from the competition page:
https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge
