"""
CodeAlpha Task 1: Credit Scoring Model - Machine Learning Engine
Provides financial data generation/loading, multi-model pipeline training,
performance benchmarking, feature importance analysis, and real-time inference.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings('ignore')


class CreditScoringEngine:
    """
    Complete Credit Scoring and Risk Assessment ML Engine.
    Trains and compares Logistic Regression, Random Forest, and Gradient Boosting models.
    """

    CATEGORICAL_COLS = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    NUMERICAL_COLS = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "debt_to_income_ratio",
        "num_open_accounts",
        "revolving_utilization"
    ]

    FEATURE_NAMES = NUMERICAL_COLS + CATEGORICAL_COLS

    def __init__(self, random_state: int = 42, dataset_size: int = 2500):
        self.random_state = random_state
        self.dataset_size = dataset_size
        
        # Initialize and build dataset
        self.df = self._generate_credit_dataset()
        self.X = self.df.drop(columns=["loan_status"])
        self.y = self.df["loan_status"]  # 0: Creditworthy / Paid, 1: High Risk / Default
        
        # Calculate feature baseline statistics
        self.feature_stats = self._compute_feature_stats()

        # Train / Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )

        # Preprocessing Pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERICAL_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS)
            ]
        )

        # Candidate Models
        self.models = {
            "LogisticRegression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", LogisticRegression(max_iter=3000, random_state=self.random_state))
            ]),
            "RandomForest": Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=self.random_state))
            ]),
            "GradientBoosting": Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=self.random_state))
            ])
        }

        self.metrics = {}
        self.roc_data = {}
        self.confusion_matrices = {}
        self.feature_importances = {}
        self.best_model_name = "GradientBoosting"

        self._train_and_evaluate_all()

    def _generate_credit_dataset(self) -> pd.DataFrame:
        """
        Synthesizes a realistic financial lending dataset modeled on standard credit bureau
        distributions (German Credit / LendingClub credit risk standards).
        """
        np.random.seed(self.random_state)
        n = self.dataset_size

        # Demographic & Employment
        person_age = np.random.gamma(shape=8.0, scale=4.2, size=n).astype(int)
        person_age = np.clip(person_age, 20, 72)

        person_income = np.random.lognormal(mean=11.1, sigma=0.55, size=n)
        person_income = np.round(np.clip(person_income, 18000, 240000), -2)

        person_emp_length = np.random.exponential(scale=5.0, size=n).astype(int)
        person_emp_length = np.clip(person_emp_length, 0, person_age - 18)

        home_options = ["RENT", "MORTGAGE", "OWN", "OTHER"]
        home_probs = [0.48, 0.40, 0.10, 0.02]
        person_home_ownership = np.random.choice(home_options, size=n, p=home_probs)

        # Loan Characteristics
        intent_options = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        intent_probs = [0.22, 0.20, 0.18, 0.16, 0.12, 0.12]
        loan_intent = np.random.choice(intent_options, size=n, p=intent_probs)

        grade_options = ["A", "B", "C", "D", "E", "F", "G"]
        grade_probs = [0.28, 0.32, 0.20, 0.11, 0.05, 0.03, 0.01]
        loan_grade = np.random.choice(grade_options, size=n, p=grade_probs)

        base_int_rates = {"A": 7.2, "B": 10.5, "C": 13.8, "D": 16.9, "E": 20.1, "F": 23.4, "G": 26.5}
        loan_int_rate = np.array([
            base_int_rates[g] + np.random.normal(0, 0.8) for g in loan_grade
        ])
        loan_int_rate = np.round(np.clip(loan_int_rate, 5.0, 29.0), 2)

        loan_amnt = np.random.gamma(shape=3.5, scale=3200, size=n)
        loan_amnt = np.round(np.clip(loan_amnt, 1000, 40000), -2)

        loan_percent_income = np.round(loan_amnt / person_income, 3)

        # Credit History & Bureau Attributes
        default_history_prob = np.where(np.isin(loan_grade, ["D", "E", "F", "G"]), 0.35, 0.08)
        cb_person_default_on_file = np.where(np.random.rand(n) < default_history_prob, "Y", "N")

        cb_person_cred_hist_length = (person_age * 0.35 + np.random.normal(0, 2, n)).astype(int)
        cb_person_cred_hist_length = np.clip(cb_person_cred_hist_length, 1, 35)

        num_open_accounts = np.random.poisson(lam=8.5, size=n)
        num_open_accounts = np.clip(num_open_accounts, 1, 28)

        revolving_utilization = np.random.beta(a=2.2, b=3.5, size=n)
        revolving_utilization = np.round(np.clip(revolving_utilization, 0.02, 0.98), 3)

        debt_to_income_ratio = np.random.beta(a=2.5, b=5.0, size=n) * 0.75
        debt_to_income_ratio = np.round(np.clip(debt_to_income_ratio, 0.05, 0.65), 3)

        # Compute Ground Truth Default Log-Odds (Risk Score Engine)
        log_odds = (
            -2.40
            + 2.8 * (loan_percent_income - 0.18)
            + 3.2 * (revolving_utilization - 0.35)
            + 2.6 * (debt_to_income_ratio - 0.25)
            + 0.12 * (loan_int_rate - 11.0)
            + 1.4 * (cb_person_default_on_file == "Y")
            + 0.5 * (person_home_ownership == "RENT")
            - 0.6 * (person_home_ownership == "OWN")
            - 0.04 * (person_emp_length - 4.5)
            - 0.03 * (cb_person_cred_hist_length - 8.0)
            + np.random.normal(0, 0.45, n)
        )
        default_prob = 1.0 / (1.0 + np.exp(-log_odds))
        loan_status = (np.random.rand(n) < default_prob).astype(int)  # 0: Good, 1: Default

        df = pd.DataFrame({
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "debt_to_income_ratio": debt_to_income_ratio,
            "num_open_accounts": num_open_accounts,
            "revolving_utilization": revolving_utilization,
            "loan_status": loan_status
        })

        return df

    def _compute_feature_stats(self) -> dict:
        """Computes baseline numerical distributions and categorical options for UI controls."""
        stats = {}
        for col in self.NUMERICAL_COLS:
            s = self.df[col]
            stats[col] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "p50": round(float(s.median()), 2),
                "p75": round(float(s.quantile(0.75)), 2),
                "good_mean": round(float(self.df[self.df["loan_status"] == 0][col].mean()), 2),
                "default_mean": round(float(self.df[self.df["loan_status"] == 1][col].mean()), 2)
            }
        
        categorical_options = {}
        for col in self.CATEGORICAL_COLS:
            categorical_options[col] = sorted(self.df[col].unique().tolist())
            
        stats["_categorical_options"] = categorical_options
        return stats

    def _train_and_evaluate_all(self):
        """Fits all ML pipelines and computes evaluation metrics, curves, and feature rankings."""
        best_auc = -1.0

        for name, pipe in self.models.items():
            pipe.fit(self.X_train, self.y_train)

            y_pred = pipe.predict(self.X_test)
            y_prob = pipe.predict_proba(self.X_test)  # col 0: Paid (Good), col 1: Default (Risk)
            y_prob_default = y_prob[:, 1]

            acc = float(accuracy_score(self.y_test, y_pred))
            auc = float(roc_auc_score(self.y_test, y_prob_default))
            prec = float(precision_score(self.y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(self.y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

            # Confusion Matrix: 0=Creditworthy, 1=Default
            cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
            self.confusion_matrices[name] = {
                "good_as_good": int(cm[0][0]),       # True Negative (Approved correctly)
                "good_as_default": int(cm[0][1]),    # False Positive (False rejection)
                "default_as_good": int(cm[1][0]),    # False Negative (Missed default / Bad loan approved)
                "default_as_default": int(cm[1][1]), # True Positive (Caught default)
                "matrix": cm.tolist()
            }

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(self.y_test, y_prob_default, pos_label=1)
            step = max(1, len(fpr) // 30)
            sampled_idx = list(range(0, len(fpr), step))
            if (len(fpr) - 1) not in sampled_idx:
                sampled_idx.append(len(fpr) - 1)

            self.roc_data[name] = {
                "fpr": [round(float(fpr[i]), 4) for i in sampled_idx],
                "tpr": [round(float(tpr[i]), 4) for i in sampled_idx],
                "auc": round(auc, 4)
            }

            self.metrics[name] = {
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "test_samples": int(len(self.y_test)),
                "default_count": int((self.y_test == 1).sum()),
                "good_count": int((self.y_test == 0).sum())
            }

            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name

        # Extract Feature Importances from Fitted Preprocessor
        transformed_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if name == "num":
                transformed_names.extend(cols)
            elif name == "cat":
                cat_names = trans.get_feature_names_out(cols).tolist()
                transformed_names.extend(cat_names)

        # Random Forest Feature Importance
        rf_model = self.models["RandomForest"].named_steps["model"]
        rf_imp = rf_model.feature_importances_
        sorted_rf = np.argsort(rf_imp)[::-1]
        self.feature_importances["RandomForest"] = [
            {"feature": transformed_names[i], "importance": round(float(rf_imp[i]), 4)}
            for i in sorted_rf[:15]
        ]

        # Gradient Boosting Feature Importance
        gb_model = self.models["GradientBoosting"].named_steps["model"]
        gb_imp = gb_model.feature_importances_
        sorted_gb = np.argsort(gb_imp)[::-1]
        self.feature_importances["GradientBoosting"] = [
            {"feature": transformed_names[i], "importance": round(float(gb_imp[i]), 4)}
            for i in sorted_gb[:15]
        ]

        # Logistic Regression Coefficients
        lr_model = self.models["LogisticRegression"].named_steps["model"]
        lr_coefs = np.abs(lr_model.coef_[0])
        sorted_lr = np.argsort(lr_coefs)[::-1]
        self.feature_importances["LogisticRegression"] = [
            {"feature": transformed_names[i], "importance": round(float(lr_coefs[i]), 4)}
            for i in sorted_lr[:15]
        ]

    def get_dataset_summary(self) -> dict:
        """Returns dataset metadata, class distribution, and feature configurations."""
        total = len(self.df)
        defaults = int((self.df["loan_status"] == 1).sum())
        good = int((self.df["loan_status"] == 0).sum())

        return {
            "total_samples": total,
            "default_count": defaults,
            "creditworthy_count": good,
            "default_rate": round((defaults / total) * 100, 2),
            "approval_rate": round((good / total) * 100, 2),
            "numerical_features": self.NUMERICAL_COLS,
            "categorical_features": self.CATEGORICAL_COLS,
            "best_model": self.best_model_name,
            "feature_stats": self.feature_stats
        }

    def get_preset_sample(self, preset_type: str = "prime") -> dict:
        """
        Returns a curated loan applicant scenario:
        'prime', 'subprime', 'borderline', or 'random'.
        """
        if preset_type == "prime":
            # High income, low DTI, low utilization, A grade, no default
            sample_data = {
                "person_age": 38,
                "person_income": 115000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 9,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "loan_amnt": 15000.0,
                "loan_int_rate": 7.5,
                "loan_percent_income": 0.13,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 14,
                "debt_to_income_ratio": 0.18,
                "num_open_accounts": 11,
                "revolving_utilization": 0.16
            }
            true_status = "Creditworthy / Approved"
        elif preset_type == "subprime":
            # Low income, high DTI, high utilization, past default, E grade
            sample_data = {
                "person_age": 25,
                "person_income": 32000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 1,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "E",
                "loan_amnt": 18000.0,
                "loan_int_rate": 21.5,
                "loan_percent_income": 0.56,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 3,
                "debt_to_income_ratio": 0.58,
                "num_open_accounts": 6,
                "revolving_utilization": 0.89
            }
            true_status = "High Risk / Default Likely"
        elif preset_type == "borderline":
            # Moderate income, medium DTI (~0.35), C grade
            best_pipe = self.models[self.best_model_name]
            probs = best_pipe.predict_proba(self.X_test)[:, 1]
            diff = np.abs(probs - 0.5)
            closest_idx = np.argmin(diff)
            sample_idx = self.X_test.index[closest_idx]
            row = self.X.loc[sample_idx].to_dict()
            sample_data = {k: (float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in row.items()}
            true_num = int(self.y.loc[sample_idx])
            true_status = "High Risk / Default Likely" if true_num == 1 else "Creditworthy / Approved"
        else:  # random
            sample_idx = np.random.choice(self.X_test.index)
            row = self.X.loc[sample_idx].to_dict()
            sample_data = {k: (float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in row.items()}
            true_num = int(self.y.loc[sample_idx])
            true_status = "High Risk / Default Likely" if true_num == 1 else "Creditworthy / Approved"

        return {
            "preset_type": preset_type,
            "true_status": true_status,
            "applicant": sample_data
        }

    def get_dataset_records(self, limit: int = 25, offset: int = 0, filter_label: str = None) -> dict:
        """Returns paginated applicant records for the Dataset Explorer."""
        df = self.df.copy()
        df["status_label"] = df["loan_status"].map({0: "Approved", 1: "Default Risk"})

        if filter_label and filter_label.lower() in ["approved", "default risk", "default"]:
            match_str = "Approved" if filter_label.lower() == "approved" else "Default Risk"
            df = df[df["status_label"] == match_str]

        total = len(df)
        paginated_df = df.iloc[offset: offset + limit]

        records = []
        for idx, row in paginated_df.iterrows():
            rec = {
                "id": int(idx),
                "status": row["status_label"],
                "data": {k: (round(float(v), 3) if isinstance(v, (int, float, np.number)) else str(v))
                         for k, v in row.items() if k not in ["loan_status", "status_label"]}
            }
            records.append(rec)

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": records
        }

    def predict_single(self, input_data: dict) -> dict:
        """
        Performs real-time credit risk inference across all 3 models.
        Calculates calibrated FICO-like credit score (300-850), Risk Tier,
        Risk Factor Drivers, and Model Consensus.
        """
        # Ensure correct types and derived features
        cleaned = {}
        for num_col in self.NUMERICAL_COLS:
            val = input_data.get(num_col, self.feature_stats[num_col]["mean"])
            try:
                cleaned[num_col] = float(val)
            except (ValueError, TypeError):
                cleaned[num_col] = float(self.feature_stats[num_col]["mean"])

        for cat_col in self.CATEGORICAL_COLS:
            val = input_data.get(cat_col)
            valid_opts = self.feature_stats["_categorical_options"][cat_col]
            if val not in valid_opts:
                val = valid_opts[0]
            cleaned[cat_col] = str(val)

        # Re-derive loan_percent_income if income > 0
        if cleaned["person_income"] > 0:
            cleaned["loan_percent_income"] = round(cleaned["loan_amnt"] / cleaned["person_income"], 3)

        input_df = pd.DataFrame([cleaned])

        model_results = {}
        default_votes = 0
        approve_votes = 0
        sum_default_prob = 0.0
        sum_approve_prob = 0.0

        for name, pipe in self.models.items():
            pred = int(pipe.predict(input_df)[0])  # 0: Good, 1: Default
            probs = pipe.predict_proba(input_df)[0]  # [prob_good, prob_default]
            prob_good = float(probs[0])
            prob_default = float(probs[1])

            decision = "Approved (Creditworthy)" if pred == 0 else "High Default Risk"
            confidence = prob_good if pred == 0 else prob_default

            if pred == 1:
                default_votes += 1
            else:
                approve_votes += 1

            sum_default_prob += prob_default
            sum_approve_prob += prob_good

            model_results[name] = {
                "decision": decision,
                "is_default": (pred == 1),
                "confidence": round(confidence * 100, 2),
                "prob_approved": round(prob_good * 100, 2),
                "prob_default": round(prob_default * 100, 2)
            }

        avg_default_prob = (sum_default_prob / len(self.models)) * 100
        avg_approve_prob = (sum_approve_prob / len(self.models)) * 100

        consensus_is_default = default_votes >= 2
        consensus_decision = "Rejected (High Default Risk)" if consensus_is_default else "Approved (Creditworthy)"
        consensus_confidence = avg_default_prob if consensus_is_default else avg_approve_prob

        # Calculate Calibrated Credit Score (FICO scale: 300 to 850)
        # Higher approve probability -> higher credit score
        fico_score = int(np.clip(300 + (avg_approve_prob / 100.0) * 550, 300, 850))

        # Determine Risk Tier
        if fico_score >= 750:
            risk_tier = "Prime Borrower (Tier 1)"
            risk_badge = "success"
            recommendation = "Instant Prime Approval with lowest interest rate qualification."
        elif fico_score >= 670:
            risk_tier = "Near Prime Borrower (Tier 2)"
            risk_badge = "info"
            recommendation = "Standard Approval with regular risk verification."
        elif fico_score >= 580:
            risk_tier = "Subprime Risk (Tier 3)"
            risk_badge = "warning"
            recommendation = "Conditional Approval with higher interest margin or collateral requirement."
        else:
            risk_tier = "Deep Subprime (Tier 4)"
            risk_badge = "danger"
            recommendation = "Application Decline / High probability of credit default."

        # Risk Factors & Key Influencers
        risk_drivers = []
        if cleaned["debt_to_income_ratio"] > 0.40:
            risk_drivers.append({
                "factor": "High Debt-to-Income Ratio",
                "detail": f"{round(cleaned['debt_to_income_ratio'] * 100, 1)}% exceeds recommended 40% threshold.",
                "severity": "high"
            })
        if cleaned["revolving_utilization"] > 0.70:
            risk_drivers.append({
                "factor": "Elevated Credit Utilization",
                "detail": f"{round(cleaned['revolving_utilization'] * 100, 1)}% indicates high revolving debt stress.",
                "severity": "high"
            })
        if cleaned["cb_person_default_on_file"] == "Y":
            risk_drivers.append({
                "factor": "Historical Credit Default on File",
                "detail": "Past negative bureau record significantly elevates default probability.",
                "severity": "critical"
            })
        if cleaned["loan_percent_income"] > 0.35:
            risk_drivers.append({
                "factor": "Heavy Loan-to-Income Burden",
                "detail": f"Loan amount is {round(cleaned['loan_percent_income'] * 100, 1)}% of annual borrower income.",
                "severity": "medium"
            })
        if cleaned["person_emp_length"] < 2:
            risk_drivers.append({
                "factor": "Short Employment Duration",
                "detail": f"{int(cleaned['person_emp_length'])} year(s) of employment tenure.",
                "severity": "low"
            })

        if not risk_drivers:
            risk_drivers.append({
                "factor": "Strong Credit Profile",
                "detail": "All financial biometrics within optimal low-risk parameters.",
                "severity": "positive"
            })

        return {
            "consensus": {
                "decision": consensus_decision,
                "is_default": consensus_is_default,
                "confidence": round(consensus_confidence, 2),
                "credit_score": fico_score,
                "risk_tier": risk_tier,
                "risk_badge": risk_badge,
                "recommendation": recommendation,
                "avg_approve_prob": round(avg_approve_prob, 2),
                "avg_default_prob": round(avg_default_prob, 2),
                "agreement": f"{max(default_votes, approve_votes)}/3 Models Agreed"
            },
            "models": model_results,
            "risk_drivers": risk_drivers,
            "applicant": cleaned
        }

    def batch_predict(self, rows: list) -> dict:
        """Processes batch credit applications from JSON rows or CSV file."""
        results = []
        for i, row in enumerate(rows):
            pred_res = self.predict_single(row)
            results.append({
                "application_id": f"APP-{1000 + i + 1}",
                "decision": pred_res["consensus"]["decision"],
                "credit_score": pred_res["consensus"]["credit_score"],
                "risk_tier": pred_res["consensus"]["risk_tier"],
                "confidence": pred_res["consensus"]["confidence"],
                "is_default": pred_res["consensus"]["is_default"],
                "logistic_regression": pred_res["models"]["LogisticRegression"]["decision"],
                "random_forest": pred_res["models"]["RandomForest"]["decision"],
                "gradient_boosting": pred_res["models"]["GradientBoosting"]["decision"]
            })

        approved_cnt = sum(1 for r in results if not r["is_default"])
        rejected_cnt = sum(1 for r in results if r["is_default"])

        return {
            "total_processed": len(results),
            "approved_count": approved_cnt,
            "rejected_count": rejected_cnt,
            "approval_rate": round((approved_cnt / len(results)) * 100, 2) if results else 0,
            "results": results
        }


# Global singleton instance
ml_engine = CreditScoringEngine()
