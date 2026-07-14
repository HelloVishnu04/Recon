# 🛡️ Recon Framework

An end-to-end machine learning project for detecting and classifying cyber threats in network traffic. The system uses a trained Random Forest classifier served via a FastAPI backend, with a Flask-based web dashboard for real-time threat analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Attack Categories](#attack-categories)
- [Technologies Used](#technologies-used)

---

## 🔍 Overview

This framework leverages machine learning to analyze network traffic patterns and classify them as **Normal** or one of several cyber attack types. The system provides:

- **Automated threat detection** using trained ML models
- **Real-time classification** via REST API
- **Interactive dashboard** for security analysts
- **Confidence scoring** for each prediction
- **Actionable recommendations** based on threat type

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│              Flask Frontend (:5000)           │
│    Dashboard │ Prediction Form │ Results      │
└──────────────────┬───────────────────────────┘
                   │ HTTP Requests
                   ▼
┌──────────────────────────────────────────────┐
│            FastAPI Backend (:8000)            │
│   /api/predict │ /api/model-info │ /health    │
│         Pydantic Validation Layer             │
└──────────────────┬───────────────────────────┘
                   │ Loads
                   ▼
┌──────────────────────────────────────────────┐
│           Serialized ML Models               │
│    model.pkl │ scaler.pkl │ label_encoder.pkl │
└──────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 ML Model Training | Complete Jupyter notebook pipeline with EDA, preprocessing, training, and evaluation |
| 📊 Exploratory Data Analysis | Visualizations including correlation heatmaps, class distributions, and feature analysis |
| 🔄 Model Serialization | Pickle files for model, scaler, and label encoder for easy deployment |
| ⚡ FastAPI Backend | High-performance REST API with automatic Swagger documentation |
| ✅ Pydantic Validation | Type-safe request/response schemas with automatic validation |
| 🎨 Flask Frontend | Premium dark-themed dashboard with glassmorphism UI |
| 📈 Confidence Scoring | Probability-based confidence for each prediction |
| 🛡️ Threat Recommendations | Actionable security recommendations per threat type |

---

## 📁 Project Structure

```
Recon/
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── notebook/
│   └── recon.ipynb                   # ML training pipeline
├── models/                           # Generated after training
│   ├── model.pkl                     # Trained model
│   ├── scaler.pkl                    # Feature scaler
│   └── label_encoder.pkl            # Label encoder
├── api/
│   ├── main.py                       # FastAPI application
│   └── schemas.py                    # Pydantic schemas
└── frontend/
    ├── app.py                        # Flask application
    ├── static/
    │   └── css/
    │       └── style.css             # Premium dark theme
    └── templates/
        ├── base.html                 # Base template
        ├── index.html                # Dashboard
        ├── predict.html              # Prediction form
        └── result.html               # Results display
```

---

## 🚀 Installation

### 1. Clone & Setup

```bash
cd Recon
pip install -r requirements.txt
```

### 2. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook notebook/recon.ipynb
```

> The notebook will ask you to provide the path to your dataset CSV file.

### 3. Start the FastAPI Backend

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 4. Start the Flask Frontend

```bash
cd frontend
python app.py
```

Dashboard available at: `http://localhost:5000`

---

## 📡 API Documentation

### Health Check
```
GET /
Response: {"status": "healthy", "message": "🛡️ Recon API is running"}
```

### Model Info
```
GET /api/model-info
Response: {"model_name": "Random Forest Classifier", "features": [...], "classes": [...]}
```

### Predict Threat
```
POST /api/predict
Body: {"duration": 0, "protocol_type": "tcp", "service": "http", ...}
Response: {"prediction": "Normal", "confidence": 0.95, "threat_level": "Low", ...}
```

---

## ⚔️ Attack Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Normal** | Legitimate network traffic | Regular HTTP, FTP, SSH sessions |
| **DoS** | Denial of Service attacks | SYN flood, Smurf, Teardrop, Neptune |
| **Probe** | Surveillance/scanning attacks | Port scan, IP sweep, Nmap, Satan |
| **R2L** | Remote to Local attacks | Password guessing, Phishing, FTP write |
| **U2R** | User to Root attacks | Buffer overflow, Rootkit, Perl exploits |

---

## 🛠️ Technologies Used

- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Backend API**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Flask, Jinja2, HTML/CSS/JS
- **Serialization**: pickle
- **Notebook**: Jupyter

---

## 📝 License

This project is developed for educational and research purposes.
