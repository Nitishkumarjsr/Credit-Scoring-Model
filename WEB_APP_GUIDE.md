# CrediPulse AI &mdash; Web Application Guide
**Comprehensive Architecture, API, and Usage Documentation**

---

## 1. System Overview
**CrediPulse AI** is a production-grade automated credit underwriting and risk assessment dashboard powered by Scikit-Learn and Flask. It predicts applicant creditworthiness, computes a calibrated FICO-style credit score (300&ndash;850), assigns risk tiers (*Prime, Near Prime, Subprime, Deep Subprime*), and surfaces explainable policy risk factors.

---

## 2. Machine Learning Pipeline

### Data Attributes
The system utilizes 14 key financial and bureau parameters:
1. `person_age`: Borrower age in years.
2. `person_income`: Annual verified borrower income.
3. `person_home_ownership`: Home occupancy status (`RENT`, `MORTGAGE`, `OWN`, `OTHER`).
4. `person_emp_length`: Continuous employment tenure in years.
5. `loan_intent`: Loan purpose category (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`).
6. `loan_grade`: Underwriter credit rating grade (`A` through `G`).
7. `loan_amnt`: Requested principal loan amount in USD.
8. `loan_int_rate`: Contractual interest rate percentage.
9. `loan_percent_income`: Derived ratio of loan principal to annual income.
10. `cb_person_default_on_file`: Historical credit bureau default indicator (`Y` / `N`).
11. `cb_person_cred_hist_length`: Length of established credit history in years.
12. `debt_to_income_ratio`: Total monthly debt obligations divided by gross monthly income.
13. `num_open_accounts`: Number of active credit lines and banking facilities.
14. `revolving_utilization`: Percentage of available revolving credit currently utilized.

### Model Ensembles
- **Gradient Boosting Classifier**: 200 estimators, max depth 5, learning rate 0.08.
- **Random Forest Classifier**: 300 decision trees, max depth 12.
- **Logistic Regression**: Standardized numerical features and one-hot encoded categorical variables.

---

## 3. REST API Reference

### `GET /api/summary`
Returns dataset summary statistics, feature distributions, and baseline configurations.

### `GET /api/metrics`
Returns benchmark performance metrics, ROC curve coordinates, confusion matrices, and feature importances.

### `GET /api/sample/<preset_type>`
- Parameters: `preset_type` (`prime`, `subprime`, `borderline`, `random`)
- Returns a representative applicant profile.

### `POST /api/predict`
Evaluates a single loan application.

**Request Payload:**
```json
{
  "applicant": {
    "person_age": 32,
    "person_income": 75000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 6,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "B",
    "loan_amnt": 15000,
    "loan_int_rate": 10.5,
    "debt_to_income_ratio": 0.22,
    "revolving_utilization": 0.30,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 8,
    "num_open_accounts": 9
  }
}
```

**Response Payload:**
```json
{
  "status": "success",
  "result": {
    "consensus": {
      "decision": "Approved (Creditworthy)",
      "credit_score": 768,
      "risk_tier": "Prime Borrower (Tier 1)",
      "confidence": 92.5,
      "agreement": "3/3 Models Agreed"
    },
    "models": { ... },
    "risk_drivers": [ ... ]
  }
}
```

### `POST /api/batch-predict`
Accepts a JSON array of applicants or a `multipart/form-data` CSV file to evaluate portfolios in bulk.
