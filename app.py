"""
CodeAlpha Task 1: CrediPulse AI - Credit Scoring Web Application
Flask Production REST API Server & Microservice
Compatible with Render backend deployment & Vercel frontend cross-origin requests.
"""

import os
import io
import csv
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ml_engine import ml_engine
from db import db_manager

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB upload limit

# Enable Cross-Origin Resource Sharing (CORS) for Vercel frontend & local development
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.after_request
def set_security_headers(response):
    """Adds standard security headers to all HTTP responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Allow CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, apikey'
    return response


@app.route("/")
def index():
    """Main Web Application Dashboard (serves template if running standalone)."""
    try:
        summary = ml_engine.get_dataset_summary()
        return render_template("index.html", summary=summary)
    except Exception:
        return jsonify({
            "status": "online",
            "service": "CrediPulse AI - Credit Scoring Engine REST API",
            "version": "1.0.0",
            "docs": "/api/metrics",
            "health": "/health"
        })


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render/Kubernetes uptime monitors and frontend status badge."""
    db_health = db_manager.check_health()
    return jsonify({
        "status": "healthy",
        "service": "CrediPulse AI Backend",
        "version": "1.0.0",
        "models_loaded": {
            "logistic_regression": "LogisticRegression" in ml_engine.models,
            "random_forest": "RandomForest" in ml_engine.models,
            "gradient_boosting": "GradientBoosting" in ml_engine.models,
            "best_model": ml_engine.best_model_name
        },
        "database": db_health,
        "cors_enabled": True
    }), 200


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Returns dataset summary statistics and feature metadata."""
    try:
        data = ml_engine.get_dataset_summary()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns performance benchmarks, ROC curves, confusion matrices, and feature importances."""
    try:
        return jsonify({
            "status": "success",
            "metrics": ml_engine.metrics,
            "roc_data": ml_engine.roc_data,
            "confusion_matrices": ml_engine.confusion_matrices,
            "feature_importances": ml_engine.feature_importances,
            "best_model": ml_engine.best_model_name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample/<preset_type>", methods=["GET"])
def api_sample(preset_type):
    """Returns a realistic applicant sample case (prime, subprime, borderline, or random)."""
    try:
        if preset_type not in ["prime", "subprime", "borderline", "random"]:
            preset_type = "prime"
        sample_data = ml_engine.get_preset_sample(preset_type)
        return jsonify({"status": "success", "data": sample_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts applicant credit & financial attributes, performs real-time inference on
    Logistic Regression, Random Forest, and Gradient Boosting, and returns FICO credit score,
    risk tier, consensus decision, and risk factor drivers.
    Persists decision to Supabase / in-memory store.
    """
    try:
        req_data = request.get_json(force=True, silent=True) or {}
        applicant_data = req_data.get("applicant", req_data.get("features", req_data))

        if not isinstance(applicant_data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid input format. Expected a JSON object with applicant parameters."
            }), 400

        result = ml_engine.predict_single(applicant_data)
        
        # Persist decision to Supabase database (or in-memory cache)
        save_info = db_manager.save_application(applicant_data, result)
        result["persisted_to"] = save_info.get("saved_to", "memory")

        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent underwriting decisions from Supabase / audit store."""
    try:
        limit = int(request.args.get("limit", 30))
        limit = max(1, min(limit, 100))
        records = db_manager.get_recent_applications(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(records),
            "storage": "supabase" if db_manager.is_configured else "in-memory-cache",
            "data": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Returns portfolio underwriting analytics (approval rate, avg FICO score, risk tiers)."""
    try:
        analytics = db_manager.get_analytics_summary()
        return jsonify({
            "status": "success",
            "storage": "supabase" if db_manager.is_configured else "in-memory-cache",
            "data": analytics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/dataset/records", methods=["GET"])
def api_dataset_records():
    """Returns paginated records from the synthetic credit bureau dataset."""
    try:
        limit = int(request.args.get("limit", 25))
        offset = int(request.args.get("offset", 0))
        filter_label = request.args.get("filter", None)

        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        records_data = ml_engine.get_dataset_records(limit=limit, offset=offset, filter_label=filter_label)
        return jsonify({"status": "success", "data": records_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Processes batch credit applications from JSON array or uploaded CSV file."""
    try:
        rows = []
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected."}), 400

            content = file.stream.read().decode("utf-8", errors="replace")
            stream = io.StringIO(content, newline=None)
            csv_reader = csv.DictReader(stream)
            for row in csv_reader:
                cleaned_row = {}
                for k, v in row.items():
                    if k:
                        cleaned_row[k.strip()] = v.strip()
                if cleaned_row:
                    rows.append(cleaned_row)
                    if len(rows) > 500:
                        break
        else:
            req_data = request.get_json(force=True, silent=True) or {}
            rows = req_data.get("rows", req_data.get("applicants", []))
            if isinstance(rows, list):
                rows = rows[:500]

        if not rows:
            return jsonify({"status": "error", "message": "No valid data rows provided for batch inference."}), 400

        batch_results = ml_engine.batch_predict(rows)
        
        # Save batch job record
        db_manager.save_batch_job(batch_results.get("summary", {}), batch_results.get("results", []))

        return jsonify({"status": "success", "data": batch_results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("\n========================================================")
    print("CrediPulse AI: Credit Scoring REST API Server Running")
    print(f"Local Server: http://127.0.0.1:{port}")
    print(f"Health Check: http://127.0.0.1:{port}/health")
    print("========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
