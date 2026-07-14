"""
FastAPI Backend for Recon Framework

Provides REST API endpoints for:
- Health check
- Model information
- Threat prediction from network traffic features
"""

import os
import sys
import pickle
import json

# Force UTF-8 encoding for stdout and stderr to prevent Windows UnicodeEncodeErrors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    NetworkTrafficInput,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ErrorResponse
)

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="🛡️ Recon API",
    description="REST API for detecting and classifying cyber threats in network traffic using Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global Variables for Model Artifacts
# ============================================================

model = None
scaler = None
label_encoder = None
feature_encoders = None
metadata = None

# ============================================================
# Load Model Artifacts on Startup
# ============================================================

MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
)
# If relative path, resolve against project root
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODELS_DIR)


def load_model_artifacts():
    """Load all pickle files and metadata on startup."""
    global model, scaler, label_encoder, feature_encoders, metadata

    try:
        # Load trained model
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")

        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {scaler_path}")

        # Load label encoder
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label Encoder loaded from {encoder_path}")

        # Load feature encoders
        feature_encoders_path = os.path.join(MODELS_DIR, "feature_encoders.pkl")
        with open(feature_encoders_path, "rb") as f:
            feature_encoders = pickle.load(f)
        print(f"✅ Feature Encoders loaded from {feature_encoders_path}")

        # Load metadata
        metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded from {metadata_path}")

        print(f"\n🎉 All model artifacts loaded successfully!")
        print(f"   Classes: {metadata['target_classes']}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Accuracy: {metadata['accuracy']:.4f}")

    except FileNotFoundError as e:
        print(f"⚠️ Model artifacts not found: {e}")
        print(f"   Please run the training notebook first to generate pickle files.")
        print(f"   Expected directory: {MODELS_DIR}")
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")


# Load on startup
@app.on_event("startup")
async def startup_event():
    """Load model artifacts when the server starts."""
    load_model_artifacts()


# ============================================================
# Threat Level & Recommendation Logic
# ============================================================

THREAT_LEVELS = {
    "normal": "Low",
    "anomaly": "High",
    "dos": "High",
    "probe": "Medium",
    "r2l": "High",
    "u2r": "Critical",
}

RECOMMENDATIONS = {
    "normal": "✅ Traffic appears normal. Continue standard monitoring and maintain security protocols.",
    "anomaly": "🚨 Anomaly detected! Network traffic pattern deviates significantly from typical profile. Inspect connection features, check for potential scan or flood patterns, and check system logs.",
    "dos": "🚨 Denial of Service attack detected! Immediately activate DDoS mitigation, enable rate limiting, and notify the security team. Consider blocking the source IP.",
    "probe": "⚠️ Reconnaissance/scanning activity detected. Review firewall rules, close unnecessary ports, and monitor for follow-up attacks.",
    "r2l": "🔴 Remote to Local attack detected! Review and strengthen authentication mechanisms, check for unauthorized access, and audit user privileges.",
    "u2r": "🔴 User to Root privilege escalation detected! This is CRITICAL. Immediately isolate the affected system, review all user activities, and perform a full security audit.",
}


def get_threat_level(prediction: str) -> str:
    """Get threat severity level for a prediction class."""
    pred_lower = prediction.lower()
    for key, level in THREAT_LEVELS.items():
        if key in pred_lower:
            return level
    return "Medium"


def get_recommendation(prediction: str) -> str:
    """Get security recommendation for a prediction class."""
    pred_lower = prediction.lower()
    for key, rec in RECOMMENDATIONS.items():
        if key in pred_lower:
            return rec
    return "⚠️ Unknown threat type detected. Investigate immediately and escalate to the security team."


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API status."""
    return HealthResponse(
        status="healthy",
        message="🛡️ Recon API is running",
        model_loaded=model is not None
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded ML model."""
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training notebook first."
        )

    return ModelInfoResponse(
        model_name="Random Forest Classifier",
        n_estimators=metadata["n_estimators"],
        n_features=metadata["n_features"],
        n_classes=metadata["n_classes"],
        feature_names=metadata["feature_names"],
        target_classes=metadata["target_classes"],
        accuracy=metadata["accuracy"],
        precision=metadata["precision"],
        recall=metadata["recall"],
        f1_score=metadata["f1_score"],
        training_samples=metadata["training_samples"],
        testing_samples=metadata["testing_samples"]
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_threat(input_data: NetworkTrafficInput):
    """
    Predict cyber threat type from network traffic features.

    Accepts a dictionary of network traffic features and returns:
    - Predicted threat class
    - Confidence score
    - Threat severity level
    - All class probabilities
    - Security recommendation
    """
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training notebook first to generate pickle files."
        )

    try:
        # Get feature names from metadata
        expected_features = metadata["feature_names"]
        categorical_columns = metadata.get("categorical_columns", [])
        input_features = input_data.features

        # Build feature vector in the correct order
        feature_vector = {}
        for feature_name in expected_features:
            if feature_name in input_features:
                value = input_features[feature_name]

                # Encode categorical features
                if feature_name in categorical_columns and feature_encoders:
                    if feature_name in feature_encoders:
                        encoder = feature_encoders[feature_name]
                        try:
                            value = encoder.transform([str(value)])[0]
                        except ValueError:
                            # Unknown category - use 0 as default
                            value = 0
                    else:
                        value = 0

                feature_vector[feature_name] = float(value)
            else:
                feature_vector[feature_name] = 0.0

        # Create DataFrame with proper feature order
        input_df = pd.DataFrame([feature_vector], columns=expected_features)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Decode prediction
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Build probability dictionary
        all_probabilities = {}
        for idx, class_name in enumerate(label_encoder.classes_):
            all_probabilities[class_name] = round(float(prediction_proba[idx]), 4)

        # Get confidence (max probability)
        confidence = round(float(max(prediction_proba)), 4)

        # Get threat level and recommendation
        threat_level = get_threat_level(prediction_label)
        recommendation = get_recommendation(prediction_label)

        return PredictionResponse(
            success=True,
            prediction=prediction_label,
            confidence=confidence,
            threat_level=threat_level,
            all_probabilities=all_probabilities,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/api/features", tags=["Model"])
async def get_feature_names():
    """Get the list of expected feature names for prediction input."""
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training notebook first."
        )

    return {
        "feature_names": metadata["feature_names"],
        "categorical_columns": metadata.get("categorical_columns", []),
        "total_features": metadata["n_features"]
    }


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
