# 💳 CrediPulse AI &mdash; Intelligent Credit Scoring & Loan Underwriting
**CodeAlpha Machine Learning Internship &bull; Task 1**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application, automated underwriting engine, and comparative credit risk analysis system.

The project trains and evaluates **Logistic Regression**, **Random Forest (300 trees)**, and **Gradient Boosting**, serving them through a modern glassmorphic loan simulator dashboard and REST API.

---

## 🌟 Web Application Features

- **Interactive Loan Simulator Studio**: 14 financial and bureau sliders/selectors (*Income, DTI, Loan Amount, Home Ownership, Credit History, Revolving Utilization, Open Accounts, Past Default*) with real-time underwriting consensus.
- **Calibrated FICO-Equivalent Score (300&ndash;850)**: Probability-calibrated credit rating and Risk Tier classification (*Prime, Near Prime, Subprime, Deep Subprime*).
- **Automated Underwriting Policy Engine**: Real-time identification of critical risk drivers (e.g. DTI > 40%, high revolving utilization > 70%, historical defaults).
- **Curated Applicant Presets**: 1-click loading for *Prime Applicant*, *Subprime Risk*, *Borderline Case*, and *Random Profile*.
- **Model Evaluation Dashboard**: Multi-model ROC curves (interactive Chart.js), confusion matrix heatmaps, and Gini feature importance rankings.
- **Credit Bureau Dataset Explorer**: Paginated and searchable table of 2,500 modeled financial records with instant "Load into Studio" capability.
- **Batch Underwriting Processor**: Multi-applicant batch inference with CSV upload and downloadable underwriting reports.
- **Developer REST API Playground**: Interactive API tester with cURL, Python, and JavaScript snippets.

---

## 📁 Repository Structure

```
CodeAlpha_CreditScoringModel/
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python package dependencies
├── app.py                            # Flask web server & secure REST API
├── ml_engine.py                      # ML model training, evaluation & inference pipeline
├── run_web_app.py                    # One-click browser launcher
├── credit_scoring.py                 # Standalone CLI training & evaluation script
├── credit_scoring_model.joblib       # Serialized top-performing ML pipeline
├── model_results.csv                 # Baseline benchmark metrics
├── roc_curve.png                     # Baseline ROC curve chart
├── templates/
│   └── index.html                    # Dashboard user interface
├── static/
│   ├── css/
│   │   └── style.css                 # FinTech glassmorphism dark theme styling
│   └── js/
│       └── app.js                    # Frontend interactivity & charts
├── WEB_APP_GUIDE.md                  # Comprehensive web application guide
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
git clone <your-github-repo-url>
cd CodeAlpha_CreditScoringModel
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application

**Option A &mdash; One-Click Launch (Auto-opens browser):**
```bash
python run_web_app.py
```

**Option B &mdash; Direct Flask Server:**
```bash
python app.py
```
Open **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.

---

## 💻 Standalone CLI Script

To run standalone model training, evaluation, and benchmark generation:
```bash
python credit_scoring.py
```
Or specify a custom CSV dataset:
```bash
python credit_scoring.py --data your_dataset.csv --target loan_status
```

---

## 📊 Model Performance Comparison

| Model | Accuracy | ROC-AUC | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **94.20%** | **0.9680** | **0.9415** | **0.9420** | **0.9412** |
| **Random Forest (300 Trees)** | **93.80%** | **0.9635** | **0.9370** | **0.9380** | **0.9372** |
| **Logistic Regression (StandardScaler)** | **89.60%** | **0.9140** | **0.8930** | **0.8960** | **0.8925** |

---

## 🔒 Security & Privacy Practices

- **Zero External Data Leakage**: All ML inference and financial scoring run completely locally within the application instance.
- **HTTP Security Headers**: Implements `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, and `Referrer-Policy`.
- **DoS & Payload Protection**: File upload size constrained (`MAX_CONTENT_LENGTH = 16MB`) and batch limits enforced.
- **Sanitized Inputs**: Safe character decoding, numerical casting, and validation on all API endpoints.

---

## 📜 License
This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
