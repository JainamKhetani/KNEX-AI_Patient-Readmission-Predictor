# ml_train.py
# RUN WITH: python ml_train.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
import xgboost as xgb
import shap

os.makedirs("models", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

print("=" * 60)
print("  ML TRAINING PIPELINE")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("data/processed/silver_clean.csv")
print(f"  Shape: {df.shape}")

# ── Feature engineering ────────────────────────────────────────────────
print("\n[2] Feature engineering...")

# Encode categorical columns to numbers
cat_cols = ["race", "gender", "insulin", "diabetes_med",
            "change_flag", "a1c_result"]
cat_cols = [c for c in cat_cols if c in df.columns]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
    encoders[col] = le

# Final feature list for ML
feature_cols = [
    "age_numeric", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_diagnoses",
    "number_outpatient", "number_emergency", "number_inpatient",
    "admission_type_id", "discharge_disposition_id"
] + [c + "_enc" for c in cat_cols]

feature_cols = [c for c in feature_cols if c in df.columns]
print(f"  Features used: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

df_ml = df[feature_cols + ["readmitted_30"]].dropna()
X = df_ml[feature_cols]
y = df_ml["readmitted_30"]

print(f"  Dataset size: {len(X):,}")
print(f"  Class balance: {y.mean()*100:.1f}% positive (readmitted)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── MODEL 1: XGBoost (primary model) ──────────────────────────────────
print("\n[3] Training XGBoost classifier...")

# scale_pos_weight handles class imbalance (88% negative, 12% positive)
scale = (y == 0).sum() / (y == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    use_label_encoder=False,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred  = xgb_model.predict(X_test)
y_prob  = xgb_model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_prob)

print(f"\n  XGBoost Results:")
print(f"  ROC-AUC       : {auc:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Readmitted", "Readmitted"])
disp.plot(ax=ax, colorbar=False)
ax.set_title(f"XGBoost — Confusion Matrix (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig("data/plots/confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: data/plots/confusion_matrix.png")

joblib.dump(xgb_model, "models/xgb_readmission.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
print("  ✅ XGBoost model saved to models/xgb_readmission.pkl")

# ── SHAP explainability ────────────────────────────────────────────────
print("\n[4] Generating SHAP explanations...")
explainer    = shap.TreeExplainer(xgb_model)
shap_sample  = X_test.iloc[:500]
shap_values  = explainer.shap_values(shap_sample)

# SHAP summary bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — 30-day Readmission")
plt.tight_layout()
plt.savefig("data/plots/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP dot plot (shows direction of effect)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot — Feature Impact Direction")
plt.tight_layout()
plt.savefig("data/plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ SHAP plots saved to data/plots/")

# ── MODEL 2: Logistic Regression (interpretable baseline) ─────────────
print("\n[5] Training Logistic Regression (baseline)...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_sc, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1])
print(f"  Logistic Regression AUC: {lr_auc:.4f}")
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("  ✅ LR model saved")

# ── MODEL 3: K-Means clustering (patient segmentation) ────────────────
print("\n[6] K-Means patient segmentation...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

df_ml = df_ml.copy()
df_ml["cluster"] = cluster_labels

cluster_profile = df_ml.groupby("cluster").agg(
    patients=("readmitted_30", "count"),
    readmit_rate=("readmitted_30", "mean"),
    avg_los=("time_in_hospital", "mean"),
    avg_medications=("num_medications", "mean"),
    avg_diagnoses=("number_diagnoses", "mean")
).round(3)

print("\n  Cluster profiles:")
print(cluster_profile.to_string())
joblib.dump(kmeans, "models/kmeans_segments.pkl")
print("  ✅ K-Means model saved")

# ── MODEL 4: Length of stay regression ────────────────────────────────
print("\n[7] Training LOS regression model...")
los_model = Ridge(alpha=1.0)
los_model.fit(X_train, df.loc[X_train.index, "time_in_hospital"]
              if "time_in_hospital" in df.columns
              else y_train)
joblib.dump(los_model, "models/los_ridge.pkl")
print("  ✅ LOS Ridge model saved")

# ── Save predictions to CSV for dashboard use ─────────────────────────
print("\n[8] Saving predictions...")
df_pred = df_ml.copy()
df_pred["risk_score"]    = xgb_model.predict_proba(X[feature_cols])[:, 1]
df_pred["risk_level"]    = pd.cut(df_pred["risk_score"],
                                   bins=[0, 0.3, 0.5, 0.7, 1.0],
                                   labels=["Low", "Moderate", "High", "Critical"])
df_pred["cluster"]       = cluster_labels
df_pred.to_csv("data/processed/predictions.csv", index=False)
print("  ✅ Predictions saved to data/processed/predictions.csv")

# ── Final summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ML TRAINING COMPLETE")
print(f"  XGBoost AUC         : {auc:.4f}")
print(f"  Logistic Reg AUC    : {lr_auc:.4f}")
print(f"  Patient clusters    : 4")
print(f"  Models saved to     : models/")
print(f"  Plots saved to      : data/plots/")
print(f"{'='*60}")