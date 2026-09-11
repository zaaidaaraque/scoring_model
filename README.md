# Credit Scoring Model
This project contains my solution for building a **credit scorecard model** to predict the probability of default of credit card applicants.
 
The objective is to build a classic credit risk scorecard, including reject inference for previously rejected applicants, and to use it to score new client applications.
 
The project covers the complete credit scoring workflow, from exploratory data analysis and preprocessing to optimal binning, Weight of Evidence (WOE) encoding, scorecard modeling and reject inference.
 
---
 
## Project Structure
 
```
.
├── notebook_scoring_model.ipynb 
├── data_scoring.csv
└── README.md
```
 
---
 
## Dataset
 
The dataset (`data_scoring.csv`) contains information on credit card applicants, split into three groups:
 
* **accepted**: applicants who received a credit card (`Cardhldr = 1`).
* **rejected**: applicants who did not receive a credit card (`Cardhldr = 0`).
* **new_clients**: applicants pending a decision (`Cardhldr` missing).
Each observation contains:
 
- ID: Unique applicant identifier.
- Cardhldr: Whether the applicant received a card (1), was rejected (0), or is a new applicant (missing).
- default: Target variable — 1 → defaulted, 0 → did not default. Only available for accepted applicants.
- Age: Applicant's age.
- Income: Applicant's income.
- Exp_Inc: Ratio of credit card expenditure to income.
- Avgexp: Average monthly credit card expenditure.
- Ownrent: Whether the applicant owns or rents their home.
- Selfempl: Whether the applicant is self-employed.
- Depndt: Number of dependents.
- Inc_per: Income per dependent.
- Cur_add: Time (months) at current address.
- Major: Whether the applicant holds a major credit card.
- Active: Number of active credit accounts.
---
 
## Methodology
The project follows the typical workflow of a credit scoring pipeline:
 
### 1. Exploratory Data Analysis (EDA)
 
The initial analysis focused on understanding the dataset and the relationship between each variable and default risk.
 
The following aspects were explored:
 
- Dataset structure, data types and missing values.
- Target class distribution (imbalanced: 89.54% non-default vs. 10.46% default).
- Distribution of each feature and its relationship with default.
- Identification of anomalous values (e.g. `Age` below 18).
This analysis helped identify data quality issues and define the preprocessing strategy.
 
### 2. Preprocessing
 
The preprocessing pipeline includes:
 
- Conversion of invalid `Age` values (< 18) to NaN, treated as anomalous.
- Stratified train/test split (80/20) on the `accepted` applicants.
- Imputation of missing `Age` values with the median calculated on train only, applied to test, rejected and new clients.
- Removal of `Exp_Inc` and `Avgexp` from the modeling variables, since both are structurally zero for rejected and new applicants (they never held a card) and would bias the reject-inference step.
- Optimal binning of numerical and categorical variables using `optbinning`, with variable selection based on Information Value (IV).

### 3. Model
 
- **Stage 1 — Accepted-only scorecard**: a Logistic Regression scorecard (via `optbinning.Scorecard`, scaled with the `pdo_odds` method) was fitted on accepted applicants only.
- **Reject inference**: the Stage 1 scorecard was applied to the rejected applicants, using the probability cutoff that maximizes F1-score on the accepted training set, to infer a `default` label for them.
- **Stage 2 — Full scorecard**: a new scorecard was refitted on accepted + inferred-rejected applicants.
- **Scoring new clients**: the final scorecard was applied to the `new_clients` group to estimate their probability and predicted class, comparing the estimate with and without reject inference.

---
 
## Key Findings
 
Some of the main conclusions obtained during the project include:
 
- `Age` presents anomalous values below 18, all corresponding to non-defaulted customers, and required cleaning before modeling.
- `Active` (number of active credit accounts) is by far the strongest predictor of default, showing the highest Information Value.
- `Cur_add`, `Age`, `Inc_per` and `Income` also show a clear relationship with default risk, while `Depndt`, `Ownrent` and `Selfempl` show limited discriminative power.
- `Exp_Inc` and `Avgexp` are structurally zero for rejected and new applicants (they never held a card), so they were excluded from the model to avoid biasing reject inference.
- Reject inference classified the large majority of rejected applicants as low-risk (non-default), consistent across both inference passes (~96–98%).
- Including reject inference increased the estimated average default probability for new clients (from 11.7% to 14.5%), reflecting a more realistic population-level risk once rejected applicants are accounted for.
---
 
## Stack
 
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OptBinning
---
 
## How to run
Clone the repository:
 
```bash
git clone https://github.com/zaaidaaraque/master-data-science-business-analytics.git
cd master-data-science-business-analytics/credit_scoring
```
 
Install the required libraries:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn optbinning
```
 
Launch Jupyter Notebook:
 
```bash
jupyter notebook notebook_scoring_model.ipynb
```
 
