"""
Flask Frontend for Recon Framework

Serves the web dashboard and communicates with the FastAPI backend
for threat predictions.
"""

import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import requests
import json

# Force UTF-8 encoding for stdout and stderr to prevent Windows UnicodeEncodeErrors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "recon-secret-key-2024")

# FastAPI Backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ============================================================
# Helper Functions
# ============================================================

def get_model_info():
    """Fetch model information from FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass
    return None


def get_feature_names():
    """Fetch feature names from FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/features", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass
    return None


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Analyze Traffic / Main Page."""
    feature_data = get_feature_names()
    model_info = get_model_info()
    api_status = "Online" if model_info else "Offline"

    return render_template(
        "predict.html",
        feature_data=feature_data,
        model_info=model_info,
        api_status=api_status
    )


@app.route("/predict", methods=["GET"])
def predict_page():
    """Redirect to main index page."""
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction form submission."""
    try:
        # Get feature data to know categorical columns
        feature_data = get_feature_names()
        categorical_cols = feature_data.get("categorical_columns", []) if feature_data else []

        # Collect form data into features dictionary
        features = {}
        for key, value in request.form.items():
            if key.startswith("feature_"):
                feature_name = key.replace("feature_", "", 1)
                # Try to convert to float for numerical features
                if feature_name not in categorical_cols:
                    try:
                        features[feature_name] = float(value) if value else 0.0
                    except ValueError:
                        features[feature_name] = value
                else:
                    features[feature_name] = value

        # Send to FastAPI backend
        payload = {"features": features}
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return render_template("result.html", result=result, features=features)
        else:
            error_detail = response.json().get("detail", "Unknown error occurred")
            flash(f"Prediction Error: {error_detail}", "error")
            return redirect(url_for("index"))

    except requests.exceptions.ConnectionError:
        flash("❌ Cannot connect to the API server. Please make sure FastAPI is running on port 8000.", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"❌ Error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    """Redirect to index."""
    return redirect(url_for("index"))


# ============================================================
# Run Flask App
# ============================================================

if __name__ == "__main__":
    flask_host = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port = int(os.getenv("FLASK_PORT", 5000))
    flask_debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

    print("\n" + "=" * 60)
    print("🛡️  Recon - Web Dashboard")
    print("=" * 60)
    print(f"\n🌐 Dashboard:  http://localhost:{flask_port}")
    print(f"📡 API Server: {API_BASE_URL}")
    print(f"\n⚠️  Make sure FastAPI is running: uvicorn main:app --port 8000")
    print("=" * 60 + "\n")

    app.run(debug=flask_debug, host=flask_host, port=flask_port)
