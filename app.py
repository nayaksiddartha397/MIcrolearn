"""
app.py
======
Loan repayment predictor — Flask REST API.
Dataset : Lending Club

Routes
------
    GET  /              →  static/index.html
    POST /api/predict   →  run microlearn pipeline, return JSON
    GET  /api/metrics   →  return saved test-set metrics
"""

import os, sys, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
bundle = None


def load_bundle():
    global bundle
    if not os.path.exists(MODEL_PATH):
        print("  ⚠  model/model.pkl not found.")
        print("     Run  python train.py  first, then restart.\n")
        return
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    m = bundle["test_metrics"]
    print(f"  ✓  Model loaded  |  {len(bundle['all_cols'])} features  "
          f"|  ACC {m['accuracy']:.3f}  |  AUC {m['roc_auc']:.3f}")


# ── feature engineering (must mirror train.py exactly) ───────────────────────

def engineer_features(data: dict) -> dict:
    """Derive the four engineered features from raw form inputs."""
    ann_inc     = max(float(data.get("annual_inc",  50_000)), 1.0)
    loan_amnt   = float(data.get("loan_amnt",   10_000))
    installment = float(data.get("installment",    300))

    data["loan_amnt_to_annual_inc"]    = loan_amnt / ann_inc
    data["installment_to_monthly_inc"] = installment / (ann_inc / 12.0 + 1e-9)
    data["log_annual_inc"]             = float(np.log1p(ann_inc))

    if ann_inc < 30_000:
        data["income_category"] = "low"
    elif ann_inc < 60_000:
        data["income_category"] = "medium"
    else:
        data["income_category"] = "high"

    return data


def build_feature_row(data: dict) -> np.ndarray:
    """Ordered (1 × n_features) object array ready for the pipeline."""
    row = []
    for col in bundle["all_cols"]:
        val = data.get(col)
        if val is None:
            val = 0.0 if col in bundle["numeric_cols"] else "Unknown"
        row.append(val)
    return np.array([row], dtype=object)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if bundle is None:
        return jsonify({"error": "Model not loaded. Run python train.py first."}), 503

    data     = engineer_features(dict(request.get_json(force=True)))
    X        = build_feature_row(data)
    pipeline = bundle["pipeline"]

    pred   = int(pipeline.predict(X)[0])
    proba  = float(pipeline.predict_proba(X)[0][1])   # P(default)

    if proba < 0.35:
        risk_level, risk_color = "low",    "#22c55e"
    elif proba < 0.60:
        risk_level, risk_color = "medium", "#f59e0b"
    else:
        risk_level, risk_color = "high",   "#f43f5e"

    # top-5 numeric feature importances with live input values
    importances = bundle["importances"]
    feat_names  = bundle["feat_names"]
    n_num       = len(bundle["numeric_cols"])
    top5        = np.argsort(importances[:n_num])[::-1][:5]
    top_factors = [
        {
            "feature":    feat_names[i],
            "label":      feat_names[i].replace("_", " ").title(),
            "importance": round(float(importances[i]), 4),
            "value":      round(float(data.get(feat_names[i], 0)), 3),
        }
        for i in top5
    ]

    return jsonify({
        "prediction":  pred,
        "probability": round(proba, 4),
        "verdict":     "DENIED"   if pred == 1 else "APPROVED",
        "risk_level":  risk_level,
        "risk_color":  risk_color,
        "top_factors": top_factors,
    })


@app.route("/api/metrics")
def get_metrics():
    if bundle is None:
        return jsonify({"error": "Model not loaded."}), 503

    m           = bundle["test_metrics"]
    importances = bundle["importances"]
    feat_names  = bundle["feat_names"]
    top10       = np.argsort(importances)[::-1][:10]
    top_features = [
        {
            "name":       feat_names[i] if i < len(feat_names) else f"feat_{i}",
            "importance": round(float(importances[i]), 4),
        }
        for i in top10
    ]
    return jsonify({
        "accuracy":         round(m["accuracy"], 4),
        "roc_auc":          round(m["roc_auc"],  4),
        "confusion_matrix": m["confusion_matrix"],
        "report":           m["report"],
        "top_features":     top_features,
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  microlearn  ·  Loan Repayment Predictor  ·  app.py")
    print("=" * 56 + "\n")
    load_bundle()
    print("\n  Open  →  http://localhost:5000\n")
    app.run(debug=True, port=5000)