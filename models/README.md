# Models Directory

This directory will contain the trained model artifacts after running the Jupyter notebook:

- `model.pkl` — Trained Random Forest Classifier
- `scaler.pkl` — Fitted StandardScaler
- `label_encoder.pkl` — Fitted LabelEncoder for target classes
- `feature_encoders.pkl` — Fitted LabelEncoders for categorical features
- `model_metadata.json` — Model metadata (features, classes, accuracy, etc.)

## How to Generate

Run the Jupyter notebook at `notebook/cyber_threat_detection.ipynb` to train the model and generate these files.
