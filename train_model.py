"""
Recon Framework
Complete Data Science Pipeline

This script:
1. Loads the KDDTest+ ARFF dataset
2. Performs Exploratory Data Analysis (EDA)
3. Preprocesses the data
4. Trains a Random Forest Classifier
5. Evaluates the model
6. Saves pickle files for deployment
"""

import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'KDDTest+.arff')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'notebook', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("  Recon Framework")
print("  Complete Data Science Pipeline")
print("=" * 70)

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)

# Parse ARFF file manually (scipy's parser is too strict for this file)
column_names = []
data_lines = []
in_data = False

with open(DATASET_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.lower() == '@data':
            in_data = True
            continue
        if in_data:
            data_lines.append(line)
        elif line.lower().startswith('@attribute'):
            # Extract attribute name: @attribute 'name' type
            parts = line.split("'")
            if len(parts) >= 2:
                col_name = parts[1]
            else:
                col_name = line.split()[1]
            column_names.append(col_name)

# Build DataFrame from CSV data lines
from io import StringIO
csv_text = '\n'.join(data_lines)
df = pd.read_csv(StringIO(csv_text), header=None, names=column_names)

print(f"\n Dataset loaded successfully!")
print(f" Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"📋 Columns: {list(df.columns)}")
print(f"\n📊 First 5 rows:")
print(df.head().to_string())

# ============================================================
# STEP 2: Exploratory Data Analysis (EDA)
# ============================================================
print("\n\n" + "=" * 70)
print("📊 STEP 2: Exploratory Data Analysis (EDA)")
print("=" * 70)

# 2.1 Dataset Overview
print("\n--- 2.1 Dataset Overview ---")
print(f"\n📏 Shape: {df.shape}")
print(f"\n📋 Data Types:")
print(df.dtypes.value_counts())

print(f"\n🔍 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values found!")
else:
    print(missing[missing > 0])

print(f"\n📈 Statistical Summary:")
print(df.describe().to_string())

# 2.2 Target Class Distribution
print("\n--- 2.2 Target Class Distribution ---")
target_column = 'class'
print(f"\n🎯 Target column: '{target_column}'")
print(f"\n📊 Class Distribution:")
print(df[target_column].value_counts())
print(f"\n📊 Class Percentage:")
print((df[target_column].value_counts(normalize=True) * 100).round(2))

# Plot: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#00d2ff', '#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3']
class_counts = df[target_column].value_counts()

axes[0].bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='white', linewidth=0.5)
axes[0].set_title('🎯 Target Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for i, (label, count) in enumerate(zip(class_counts.index, class_counts.values)):
    axes[0].text(i, count + 100, str(count), ha='center', fontweight='bold', fontsize=11)

axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors[:len(class_counts)], shadow=True, startangle=140,
            textprops={'fontsize': 12})
axes[1].set_title('📊 Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Plot saved: 01_class_distribution.png")

# 2.3 Categorical Feature Analysis
print("\n--- 2.3 Categorical Feature Analysis ---")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if target_column in categorical_cols:
    categorical_cols.remove(target_column)
print(f"  📋 Categorical features: {categorical_cols}")
for col in categorical_cols:
    print(f"\n  📊 {col}: {df[col].nunique()} unique values")
    print(f"     Top 5: {dict(df[col].value_counts().head())}")

# Plot: Protocol Type Distribution
if 'protocol_type' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, col in enumerate(categorical_cols[:3]):
        val_counts = df[col].value_counts().head(10)
        axes[idx].barh(val_counts.index, val_counts.values, color=colors[idx], edgecolor='white')
        axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Count')
        axes[idx].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_categorical_features.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Plot saved: 02_categorical_features.png")

# 2.4 Numerical Feature Analysis
print("\n--- 2.4 Numerical Feature Analysis ---")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"  📋 Numerical features: {len(numerical_cols)}")

# Plot: Top 4 Feature Distributions
top_4_by_var = df[numerical_cols].var().nlargest(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(top_4_by_var):
    ax = axes[idx // 2][idx % 2]
    ax.hist(df[col], bins=50, color=colors[idx], alpha=0.8, edgecolor='white')
    ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.suptitle('📊 Top Feature Distributions (by Variance)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_feature_distributions.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Plot saved: 03_feature_distributions.png")

# 2.5 Correlation Heatmap
print("\n--- 2.5 Correlation Analysis ---")
top_corr_features = df[numerical_cols].var().nlargest(min(20, len(numerical_cols))).index.tolist()
plt.figure(figsize=(14, 10))
correlation_matrix = df[top_corr_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
plt.title('🔥 Feature Correlation Heatmap (Top 20 by Variance)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_correlation_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Plot saved: 04_correlation_heatmap.png")

# 2.6 Feature Statistics by Class
print("\n--- 2.6 Feature Statistics by Class ---")
key_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'duration']
available_key = [f for f in key_features if f in df.columns]
if available_key:
    fig, axes = plt.subplots(1, len(available_key), figsize=(5 * len(available_key), 5))
    if len(available_key) == 1:
        axes = [axes]
    for idx, feat in enumerate(available_key):
        df.boxplot(column=feat, by=target_column, ax=axes[idx])
        axes[idx].set_title(f'{feat} by Class', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('')
    plt.suptitle('📊 Key Features by Attack Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_features_by_class.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Plot saved: 05_features_by_class.png")

# ============================================================
# STEP 3: Data Preprocessing
# ============================================================
print("\n\n" + "=" * 70)
print("⚙️  STEP 3: Data Preprocessing")
print("=" * 70)

data = df.copy()

# 3.1 Handle Missing Values
print("\n--- 3.1 Handle Missing Values ---")
missing_before = data.isnull().sum().sum()
if missing_before > 0:
    num_cols_fill = data.select_dtypes(include=[np.number]).columns
    data[num_cols_fill] = data[num_cols_fill].fillna(data[num_cols_fill].median())
    cat_cols_fill = data.select_dtypes(include=['object']).columns
    for col in cat_cols_fill:
        data[col] = data[col].fillna(data[col].mode()[0])
    print(f"  ✅ Filled {missing_before} missing values")
else:
    print("  ✅ No missing values to handle")

# 3.2 Remove Duplicates
print("\n--- 3.2 Remove Duplicates ---")
dup_count = data.duplicated().sum()
if dup_count > 0:
    data = data.drop_duplicates()
    print(f"  ✅ Removed {dup_count} duplicate rows")
    print(f"  📊 Shape after dedup: {data.shape}")
else:
    print("  ✅ No duplicate rows found")

# 3.3 Encode Categorical Features
print("\n--- 3.3 Encode Categorical Variables ---")
cat_features = data.select_dtypes(include=['object']).columns.tolist()
if target_column in cat_features:
    cat_features.remove(target_column)

label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"  ✅ Encoded '{col}': {len(le.classes_)} unique → {list(le.classes_[:5])}...")

# Encode target
target_encoder = LabelEncoder()
data[target_column] = target_encoder.fit_transform(data[target_column].astype(str))
print(f"\n🎯 Target classes: {list(target_encoder.classes_)}")
print(f"   Encoded as: {list(range(len(target_encoder.classes_)))}")

# 3.4 Separate Features and Target
print("\n--- 3.4 Separate Features & Target ---")
X = data.drop(columns=[target_column])
y = data[target_column]
feature_names = X.columns.tolist()
print(f"  📊 Features shape: {X.shape}")
print(f"  🎯 Target shape: {y.shape}")
print(f"  📋 Total features: {len(feature_names)}")

# 3.5 Feature Scaling
print("\n--- 3.5 Feature Scaling ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
print(f"  ✅ StandardScaler applied to {X_scaled.shape[1]} features")
print(f"  📊 Mean (≈0): {X_scaled.mean().mean():.6f}")
print(f"  📊 Std  (≈1): {X_scaled.std().mean():.6f}")

# 3.6 Train-Test Split
print("\n--- 3.6 Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  📊 Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  📊 Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")

# ============================================================
# STEP 4: Model Training
# ============================================================
print("\n\n" + "=" * 70)
print("🤖 STEP 4: Model Training")
print("=" * 70)

print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print(f"\n✅ Model trained successfully!")
print(f"  🌲 Number of trees: {model.n_estimators}")
print(f"  📊 Number of features: {model.n_features_in_}")
print(f"  🎯 Number of classes: {model.n_classes_}")

# ============================================================
# STEP 5: Model Evaluation
# ============================================================
print("\n\n" + "=" * 70)
print("📈 STEP 5: Model Evaluation")
print("=" * 70)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 5.1 Core Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n  🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  📊 Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  📊 Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  📊 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# 5.2 Classification Report
print(f"\n📋 CLASSIFICATION REPORT:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# 5.3 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            linewidths=0.5, linecolor='white')
plt.title('🎯 Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_confusion_matrix.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Plot saved: 06_confusion_matrix.png")

# 5.4 Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

top_n = min(15, len(feature_importance))
plt.figure(figsize=(12, 7))
top_fi = feature_importance.head(top_n)
bars = plt.barh(range(top_n), top_fi['Importance'].values, color='#00d2ff', edgecolor='white')
plt.yticks(range(top_n), top_fi['Feature'].values, fontsize=10)
plt.xlabel('Importance Score', fontsize=12)
plt.title(f'🏆 Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for bar, val in zip(bars, top_fi['Importance'].values):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_feature_importance.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Plot saved: 07_feature_importance.png")

print(f"\n📊 Top 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# 5.5 ROC Curve (if binary classification)
if model.n_classes_ == 2:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00d2ff', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#ff6b6b', lw=1, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('📈 ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_roc_curve.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ ROC AUC Score: {roc_auc:.4f}")
    print("  ✅ Plot saved: 08_roc_curve.png")

# ============================================================
# STEP 6: Save Model Artifacts (Pickle Files)
# ============================================================
print("\n\n" + "=" * 70)
print("💾 STEP 6: Saving Model Artifacts (Pickle Files)")
print("=" * 70)

# Save Model
model_path = os.path.join(MODELS_DIR, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  ✅ Model saved: {model_path}")

# Save Scaler
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✅ Scaler saved: {scaler_path}")

# Save Target Label Encoder
encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(target_encoder, f)
print(f"  ✅ Label Encoder saved: {encoder_path}")

# Save Feature Encoders
feature_encoders_path = os.path.join(MODELS_DIR, 'feature_encoders.pkl')
with open(feature_encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✅ Feature Encoders saved: {feature_encoders_path}")

# Save Metadata
metadata = {
    'feature_names': feature_names,
    'target_classes': list(target_encoder.classes_),
    'categorical_columns': cat_features,
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'n_estimators': model.n_estimators,
    'n_features': len(feature_names),
    'n_classes': len(target_encoder.classes_),
    'training_samples': int(X_train.shape[0]),
    'testing_samples': int(X_test.shape[0]),
    'dataset': 'KDDTest+.arff',
    'total_samples': int(df.shape[0])
}

metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"  ✅ Metadata saved: {metadata_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("🎉 PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n📊 Model Performance:")
print(f"  🎯 Accuracy:  {accuracy*100:.2f}%")
print(f"  📊 Precision: {precision*100:.2f}%")
print(f"  📊 Recall:    {recall*100:.2f}%")
print(f"  📊 F1-Score:  {f1*100:.2f}%")
print(f"\n🏷️  Classes: {list(target_encoder.classes_)}")
print(f"📊 Features Used: {len(feature_names)}")
print(f"\n💾 Pickle files saved in: {MODELS_DIR}")
for f_name in os.listdir(MODELS_DIR):
    f_path = os.path.join(MODELS_DIR, f_name)
    if os.path.isfile(f_path):
        f_size = os.path.getsize(f_path) / 1024
        print(f"  📄 {f_name} ({f_size:.1f} KB)")
print(f"\n📊 Plots saved in: {PLOTS_DIR}")
for f_name in sorted(os.listdir(PLOTS_DIR)):
    print(f"  🖼️  {f_name}")
print(f"\n🚀 Next Steps:")
print(f"  1. Start FastAPI:  cd api && uvicorn main:app --reload --port 8000")
print(f"  2. Start Flask:    cd frontend && python app.py")
print(f"  3. Open Dashboard: http://localhost:5000")
