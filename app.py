"""
CodeAlpha Task 1: CrediPulse AI - Credit Scoring Web Application
Flask Application & Secure REST API Server
"""

import os
import io
import csv
import json
from flask import Flask, render_template, request, jsonify
from ml_engine import ml_engine

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit


@app.after_request
def set_security_headers(response):
    """Adds standard security headers to all HTTP responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route("/")
def index():
    """Main Web Application Dashboard."""
    summary = ml_engine.get_dataset_summary()
    return render_template("index.html", summary=summary)


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
        return jsonify({"status": "success", "result": result})
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
        return jsonify({"status": "success", "data": batch_results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("\n========================================================")
    print("CrediPulse AI: Credit Scoring Web App Running!")
    print(f"Access Dashboard at: http://127.0.0.1:{port}")
    print("========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
