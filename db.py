"""
CodeAlpha Task 1: CrediPulse AI - Database Integration Layer
Connects to Supabase PostgreSQL via REST API / PostgREST.
Includes automatic in-memory fallback for local offline development.
"""

import os
import time
import json
import datetime
import requests
from dotenv import load_dotenv

# Load local environment variables if .env file exists
load_dotenv()

class SupabaseDatabaseManager:
    """
    Manages persistent storage of credit underwriting decisions,
    batch assessment jobs, and portfolio analytics on Supabase PostgreSQL.
    """

    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        
        # Local fallback store in case Supabase is not configured or offline
        self._local_history = []
        self._local_batch_jobs = []
        self._is_configured = bool(self.supabase_url and self.supabase_key)

    @property
    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def _get_headers(self) -> dict:
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    def check_health(self) -> dict:
        """Checks connection status and latency to Supabase."""
        if not self.is_configured:
            return {
                "connected": False,
                "provider": "Supabase PostgreSQL",
                "status": "not_configured",
                "message": "SUPABASE_URL and SUPABASE_KEY environment variables not set. Using local in-memory storage.",
                "local_records_count": len(self._local_history)
            }

        start_time = time.time()
        try:
            endpoint = f"{self.supabase_url}/rest/v1/credit_applications?select=id&limit=1"
            response = requests.get(endpoint, headers=self._get_headers(), timeout=4)
            latency_ms = round((time.time() - start_time) * 1000, 1)

            if response.status_code in [200, 206]:
                return {
                    "connected": True,
                    "provider": "Supabase PostgreSQL",
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "endpoint": self.supabase_url
                }
            elif response.status_code == 404:
                return {
                    "connected": False,
                    "provider": "Supabase PostgreSQL",
                    "status": "table_missing",
                    "message": "Connected to Supabase, but 'credit_applications' table is missing. Run supabase_schema.sql in Supabase SQL Editor.",
                    "latency_ms": latency_ms
                }
            else:
                return {
                    "connected": False,
                    "provider": "Supabase PostgreSQL",
                    "status": f"http_error_{response.status_code}",
                    "message": response.text[:200],
                    "latency_ms": latency_ms
                }
        except Exception as e:
            return {
                "connected": False,
                "provider": "Supabase PostgreSQL",
                "status": "connection_error",
                "message": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }

    def save_application(self, applicant_data: dict, prediction_result: dict) -> dict:
        """
        Saves a single underwriting inference decision to Supabase or local memory.
        """
        consensus = prediction_result.get("consensus", {})
        models = prediction_result.get("models", {})
        risk_drivers = prediction_result.get("risk_drivers", [])

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        record = {
            "created_at": now_iso,
            "applicant_name": applicant_data.get("applicant_name", "Applicant-" + str(int(time.time() * 1000))[-6:]),
            "person_age": int(applicant_data.get("person_age", 30)),
            "person_income": float(applicant_data.get("person_income", 0)),
            "person_home_ownership": str(applicant_data.get("person_home_ownership", "RENT")),
            "person_emp_length": float(applicant_data.get("person_emp_length", 0)),
            "loan_amnt": float(applicant_data.get("loan_amnt", 0)),
            "loan_intent": str(applicant_data.get("loan_intent", "PERSONAL")),
            "loan_grade": str(applicant_data.get("loan_grade", "B")),
            "loan_int_rate": float(applicant_data.get("loan_int_rate", 0)),
            "debt_to_income_ratio": float(applicant_data.get("debt_to_income_ratio", 0)),
            "revolving_utilization": float(applicant_data.get("revolving_utilization", 0)),
            "cb_person_cred_hist_length": float(applicant_data.get("cb_person_cred_hist_length", 0)),
            "num_open_accounts": float(applicant_data.get("num_open_accounts", 0)),
            "cb_person_default_on_file": str(applicant_data.get("cb_person_default_on_file", "N")),
            
            # Inference outputs
            "fico_score": int(consensus.get("credit_score", 650)),
            "risk_tier": str(consensus.get("risk_tier", "Moderate Risk")),
            "recommendation": str(consensus.get("recommendation", "Review")),
            "is_default_predicted": bool(consensus.get("is_default", False)),
            "consensus_probability": float(consensus.get("default_probability", 0.5)),
            "model_agreement": str(consensus.get("agreement", "N/A")),
            
            # Model details
            "lr_probability": float(models.get("LogisticRegression", {}).get("default_prob", 0)),
            "rf_probability": float(models.get("RandomForest", {}).get("default_prob", 0)),
            "gb_probability": float(models.get("GradientBoosting", {}).get("default_prob", 0)),
            "top_risk_factors": risk_drivers[:4]
        }

        # In-memory buffer
        self._local_history.insert(0, record)
        if len(self._local_history) > 200:
            self._local_history.pop()

        # Try persisting to Supabase if configured
        if self.is_configured:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/credit_applications"
                # Transform top_risk_factors for JSON column
                payload = dict(record)
                payload["top_risk_factors"] = json.dumps(record["top_risk_factors"])
                res = requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=3)
                if res.status_code in [200, 201]:
                    data = res.json()
                    return {"saved_to": "supabase", "data": data[0] if isinstance(data, list) and data else record}
            except Exception as e:
                print(f"[Supabase Sync Notice]: Could not persist to Supabase ({e}). Stored in local cache.")

        return {"saved_to": "memory", "data": record}

    def save_batch_job(self, batch_summary: dict, items: list) -> dict:
        """Saves summary of a batch loan assessment run."""
        job_record = {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_applicants": batch_summary.get("total", len(items)),
            "approved_count": batch_summary.get("approved", 0),
            "rejected_count": batch_summary.get("rejected", 0),
            "review_count": batch_summary.get("review", 0),
            "avg_credit_score": batch_summary.get("avg_score", 0),
            "approval_rate": batch_summary.get("approval_rate", 0),
            "sample_records": items[:10]
        }

        self._local_batch_jobs.insert(0, job_record)
        if len(self._local_batch_jobs) > 50:
            self._local_batch_jobs.pop()

        if self.is_configured:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/batch_underwriting_jobs"
                payload = dict(job_record)
                payload["sample_records"] = json.dumps(job_record["sample_records"])
                requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=3)
            except Exception as e:
                print(f"[Supabase Batch Notice]: {e}")

        return {"saved_to": "supabase" if self.is_configured else "memory", "summary": job_record}

    def get_recent_applications(self, limit: int = 50) -> list:
        """Fetches recent underwriting decisions from Supabase or local memory."""
        if self.is_configured:
            try:
                endpoint = f"{self.supabase_url}/rest/v1/credit_applications?select=*&order=created_at.desc&limit={limit}"
                res = requests.get(endpoint, headers=self._get_headers(), timeout=4)
                if res.status_code == 200:
                    records = res.json()
                    for r in records:
                        if isinstance(r.get("top_risk_factors"), str):
                            try:
                                r["top_risk_factors"] = json.loads(r["top_risk_factors"])
                            except Exception:
                                pass
                    return records
            except Exception as e:
                print(f"[Supabase Read Notice]: {e}")

        return self._local_history[:limit]

    def get_analytics_summary(self) -> dict:
        """Returns aggregated underwriting statistics."""
        history = self.get_recent_applications(limit=100)
        
        if not history:
            return {
                "total_decisions": 0,
                "approval_rate": 0.0,
                "avg_fico_score": 0,
                "prime_ratio": 0.0,
                "risk_tier_counts": {
                    "Prime (Very Low Risk)": 0,
                    "Near Prime (Low Risk)": 0,
                    "Moderate Risk": 0,
                    "Subprime (High Risk)": 0,
                    "Deep Subprime (Critical Risk)": 0
                }
            }

        total = len(history)
        approved = sum(1 for h in history if "Approve" in h.get("recommendation", "") or not h.get("is_default_predicted", False))
        scores = [h.get("fico_score", 0) for h in history if h.get("fico_score")]
        avg_score = round(sum(scores) / len(scores)) if scores else 0

        tier_counts = {
            "Prime (Very Low Risk)": 0,
            "Near Prime (Low Risk)": 0,
            "Moderate Risk": 0,
            "Subprime (High Risk)": 0,
            "Deep Subprime (Critical Risk)": 0
        }

        for h in history:
            tier = h.get("risk_tier", "Moderate Risk")
            if tier in tier_counts:
                tier_counts[tier] += 1
            else:
                tier_counts[tier] = 1

        return {
            "total_decisions": total,
            "approval_rate": round((approved / total) * 100, 1),
            "avg_fico_score": avg_score,
            "risk_tier_counts": tier_counts
        }

# Global singleton database manager instance
db_manager = SupabaseDatabaseManager()
