"""
Pydantic Schemas for Recon API

Defines request/response models with validation for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class NetworkTrafficInput(BaseModel):
    """
    Input schema for network traffic prediction.
    Accepts a flexible dictionary of features to support various dataset formats.
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of network traffic features. Keys should match the feature names used during training.",
        json_schema_extra={
            "example": {
                "duration": 0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 181,
                "dst_bytes": 5450,
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": 0,
                "num_failed_logins": 0,
                "logged_in": 1,
                "num_compromised": 0,
                "root_shell": 0,
                "su_attempted": 0,
                "num_root": 0,
                "num_file_creations": 0,
                "num_shells": 0,
                "num_access_files": 0,
                "is_host_login": 0,
                "is_guest_login": 0,
                "count": 8,
                "srv_count": 8,
                "serror_rate": 0.0,
                "srv_serror_rate": 0.0,
                "rerror_rate": 0.0,
                "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": 9,
                "dst_host_srv_count": 9,
                "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.11,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0,
                "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_rerror_rate": 0.0
            }
        }
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "duration": 0,
                    "protocol_type": "tcp",
                    "service": "http",
                    "flag": "SF",
                    "src_bytes": 181,
                    "dst_bytes": 5450
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for threat prediction results."""
    success: bool = Field(..., description="Whether the prediction was successful")
    prediction: str = Field(..., description="Predicted threat class (e.g., Normal, DoS, Probe, R2L, U2R)")
    confidence: float = Field(..., description="Prediction confidence score (0.0 to 1.0)", ge=0.0, le=1.0)
    threat_level: str = Field(..., description="Threat severity level: Low, Medium, High, Critical")
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probability scores for all classes"
    )
    recommendation: str = Field(..., description="Security recommendation based on the threat type")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": "Normal",
                "confidence": 0.95,
                "threat_level": "Low",
                "all_probabilities": {
                    "Normal": 0.95,
                    "DoS": 0.02,
                    "Probe": 0.01,
                    "R2L": 0.01,
                    "U2R": 0.01
                },
                "recommendation": "Traffic appears normal. Continue monitoring."
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_name: str = Field(..., description="Name of the ML model")
    n_estimators: int = Field(..., description="Number of trees in the forest")
    n_features: int = Field(..., description="Number of input features")
    n_classes: int = Field(..., description="Number of output classes")
    feature_names: List[str] = Field(..., description="List of feature names")
    target_classes: List[str] = Field(..., description="List of target class names")
    accuracy: float = Field(..., description="Model accuracy on test set")
    precision: float = Field(..., description="Weighted precision score")
    recall: float = Field(..., description="Weighted recall score")
    f1_score: float = Field(..., description="Weighted F1 score")
    training_samples: int = Field(..., description="Number of training samples")
    testing_samples: int = Field(..., description="Number of testing samples")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
