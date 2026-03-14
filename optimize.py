# optimize.py
# RUN WITH: python optimize.py

import pandas as pd
import numpy as np
import joblib
import os

print("=" * 60)
print("  OPTIMIZATION PIPELINE")
print("=" * 60)

df       = pd.read_csv("data/processed/silver_clean.csv")
features = joblib.load("models/feature_cols.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# Re-encode
cat_cols = ["race","gender","insulin","diabetes_med","change_flag","a1c_result"]
for col in cat_cols:
    if col in df.columns and col in encoders:
        df[col+"_enc"] = encoders[col].transform(
            df[col].astype(str).fillna("Unknown"))

X = df[[c for c in features if c in df.columns]].fillna(0)
y = df["readmitted_30"]

# ── Hyperparameter tuning with Optuna ─────────────────────────────────
print("\n[1] Hyperparameter tuning with Optuna...")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
            "scale_pos_weight": (y==0).sum()/(y==1).sum(),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(
            model, X, y, cv=3, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n  Best AUC  : {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Train final optimized model
    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": (y==0).sum()/(y==1).sum(),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42
    })
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, "models/xgb_optimized.pkl")
    print("  ✅ Optimized model saved to models/xgb_optimized.pkl")

except ImportError:
    print("  Optuna not installed. Run: pip install optuna")
    print("  Skipping tuning, using existing model.")

# ── Model compression — ONNX export ───────────────────────────────────
print("\n[2] ONNX model export for fast inference...")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    base_model = joblib.load("models/xgb_readmission.pkl")
    n_features = len(features)

    # Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(base_model, initial_types=initial_type)

    with open("models/readmission_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("  ✅ ONNX model saved to models/readmission_model.onnx")

    # Test ONNX inference speed
    import onnxruntime as rt
    import time
    sess     = rt.InferenceSession("models/readmission_model.onnx")
    X_sample = X.iloc[:1000].values.astype(np.float32)

    start = time.time()
    for _ in range(100):
        sess.run(None, {"float_input": X_sample})
    onnx_time = (time.time() - start) / 100 * 1000

    start = time.time()
    for _ in range(100):
        base_model.predict_proba(X_sample)
    sklearn_time = (time.time() - start) / 100 * 1000

    print(f"  ONNX inference    : {onnx_time:.2f}ms per batch")
    print(f"  sklearn inference : {sklearn_time:.2f}ms per batch")
    print(f"  Speedup           : {sklearn_time/onnx_time:.1f}x")

except Exception as e:
    print(f"  ONNX export skipped: {e}")
    print("  Run: pip install skl2onnx onnxruntime onnxmltools")

# ── Database query optimization ────────────────────────────────────────
print("\n[3] Database index optimization...")
try:
    from sqlalchemy import create_engine, text
    eng = create_engine("postgresql://admin:admin@localhost:5432/hospital_dw")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_silver_readmit ON silver.admissions(readmitted_30)",
        "CREATE INDEX IF NOT EXISTS idx_silver_patient ON silver.admissions(patient_nbr)",
        "CREATE INDEX IF NOT EXISTS idx_silver_age ON silver.admissions(age_numeric)",
        "CREATE INDEX IF NOT EXISTS idx_fact_readmit ON gold.fact_admission(readmitted_within_30)",
        "CREATE INDEX IF NOT EXISTS idx_fact_los ON gold.fact_admission(time_in_hospital)",
    ]
    with eng.connect() as conn:
        for idx in indexes:
            conn.execute(text(idx))
        conn.commit()
    print(f"  ✅ Created {len(indexes)} database indexes")
except Exception as e:
    print(f"  DB optimization skipped: {e}")

print(f"\n{'='*60}")
print(f"  OPTIMIZATION COMPLETE")
print(f"{'='*60}")