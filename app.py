"""
app.py
-------
Flask web application for the Phishing URL Detection System.

Routes:
  GET  /                -> home page with URL entry form
  POST /predict          -> analyze a URL (form submit), renders result
  POST /api/predict       -> JSON API for programmatic / real-time use
  GET  /history          -> table of past scans
  POST /history/clear    -> wipe scan history
"""

import os
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, g, redirect, url_for, flash

from feature_extraction import extract_features, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_model.joblib")
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-secret-change-me")

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run `python3 generate_dataset.py` then `python3 train_model.py` first."
    )
MODEL = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Database helpers (scan history)
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            scanned_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_history(url, prediction, confidence):
    db = get_db()
    db.execute(
        "INSERT INTO scan_history (url, prediction, confidence, scanned_at) "
        "VALUES (?, ?, ?, ?)",
        (url, prediction, confidence, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()


def get_history(limit=50):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM scan_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Core prediction logic (shared by web form + JSON API)
# ---------------------------------------------------------------------------
def predict_url(url: str):
    feats = extract_features(url)
    vector = pd.DataFrame([[feats[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    pred = MODEL.predict(vector)[0]
    proba = MODEL.predict_proba(vector)[0]

    is_phishing = bool(pred == 1)
    confidence = float(proba[1] if is_phishing else proba[0])

    result = {
        "url": url,
        "label": "Phishing" if is_phishing else "Safe",
        "is_phishing": is_phishing,
        "confidence": round(confidence * 100, 2),
        "features": feats,
    }
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    if not url:
        flash("Please enter a URL to analyze.", "warning")
        return redirect(url_for("home"))

    result = predict_url(url)
    save_history(result["url"], result["label"], result["confidence"])
    return render_template("result.html", result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for real-time / programmatic URL analysis."""
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or request.form.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    result = predict_url(url)
    save_history(result["url"], result["label"], result["confidence"])
    return jsonify(result)


@app.route("/history")
def history():
    rows = get_history()
    return render_template("history.html", rows=rows)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    db = get_db()
    db.execute("DELETE FROM scan_history")
    db.commit()
    flash("Scan history cleared.", "info")
    return redirect(url_for("history"))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    # Ensure DB exists even when imported (e.g. by a WSGI server)
    init_db()