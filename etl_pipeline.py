# etl_pipeline.py
# PURPOSE: Full ETL — Extract from CSV, Transform (clean/validate),
#          Load into Bronze → Silver → Gold layers in PostgreSQL.
# RUN WITH: python etl_pipeline.py

import pandas as pd
import numpy as np
import json
import os
import calendar
from datetime import datetime, date
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://admin:admin@localhost:5432/hospital_dw"
engine = create_engine(DATABASE_URL, echo=False)

quality_issues = []
run_time = datetime.now()

def log_quality(step, table, check, total, failed, details=""):
    pass_rate = round((total - failed) / total * 100, 2) if total > 0 else 0
    status = "PASS" if failed == 0 else ("WARN" if pass_rate > 90 else "FAIL")
    quality_issues.append({
        "step_name": step, "table_name": table, "check_name": check,
        "records_total": total, "records_failed": failed,
        "pass_rate_pct": pass_rate, "status": status, "details": details
    })
    icon = "✅" if status == "PASS" else ("⚠️" if status == "WARN" else "❌")
    print(f"  {icon} [{status}] {check}: {failed}/{total} issues "
          f"({pass_rate}% pass) {details}")

print("=" * 60)
print("  ETL PIPELINE — PATIENT READMISSION DATA")
print(f"  Started: {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# DROP ALL GOLD TABLES WITH CASCADE UPFRONT — prevents all FK errors
# ══════════════════════════════════════════════════════════════════════
print("\n[SETUP] Dropping existing Gold tables to allow clean reload...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS gold.fact_admission CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS gold.dim_patient CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS gold.dim_diagnosis CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS gold.dim_date CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS gold.dim_admission_type CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS gold.dim_discharge CASCADE"))
    conn.commit()
print("  ✅ Gold tables cleared")

# ══════════════════════════════════════════════════════════════════════
# EXTRACT — Load raw CSV file
# ══════════════════════════════════════════════════════════════════════
print("\n[EXTRACT] Loading raw dataset...")

df = pd.read_csv(
    "data/raw/diabetic_data.csv",
    na_values=["?", "Unknown/Invalid", "NULL", "null", ""],
    dtype=str
)

print(f"  Rows loaded    : {len(df):,}")
print(f"  Columns loaded : {len(df.columns)}")
print(f"  Columns        : {list(df.columns)}")

# ── Load to Bronze ─────────────────────────────────────────────────────
print("\n[BRONZE] Saving raw data to bronze.raw_admissions...")

def clean_for_json(row):
    """Replace NaN/float nan with None so JSON is valid for PostgreSQL."""
    return {k: (None if isinstance(v, float) and v != v else v)
            for k, v in row.items()}

bronze_records = [{
    "source_file": "diabetic_data.csv",
    "ingested_at": run_time.isoformat(),
    "record_count": len(df),
    "raw_json": json.dumps(clean_for_json(row))
} for row in df.head(1000).to_dict(orient="records")]

pd.DataFrame(bronze_records).to_sql(
    "raw_admissions", engine, schema="bronze",
    if_exists="append", index=False
)
print(f"  ✅ Bronze loaded: {len(bronze_records)} sample records")

# ══════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════
print("\n[TRANSFORM] Starting data cleaning...")
df_clean = df.copy()

# T1: Data profiling
print("\n  [T1] Data profiling")
total_rows = len(df_clean)
missing_counts = df_clean.isnull().sum()
missing_pct = (missing_counts / total_rows * 100).round(2)
print(f"  Total rows: {total_rows:,}")
print(f"  Columns with missing data:")
for col in missing_counts[missing_counts > 0].index:
    print(f"    {col:<35} {missing_counts[col]:>6} missing ({missing_pct[col]}%)")

# T2: Remove duplicates
print("\n  [T2] Removing duplicates")
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["encounter_id"])
removed = before - len(df_clean)
log_quality("T2", "silver.admissions", "Duplicate encounter_ids",
            before, removed, f"Removed {removed} duplicates")

# T3: Drop high-missing columns
print("\n  [T3] Dropping high-missing columns (>50% missing)")
high_missing = missing_pct[missing_pct > 50].index.tolist()
print(f"  Columns to drop: {high_missing}")
df_clean = df_clean.drop(columns=high_missing, errors="ignore")
log_quality("T3", "silver.admissions", "High-missing column drop",
            len(df.columns), len(high_missing),
            f"Dropped: {high_missing}")

# T4: Type conversion
print("\n  [T4] Converting data types")
numeric_cols = [
    "encounter_id", "patient_nbr", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]
conversion_errors = 0
for col in numeric_cols:
    if col in df_clean.columns:
        original = df_clean[col].copy()
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        errors = df_clean[col].isnull().sum() - original.isnull().sum()
        if errors > 0:
            conversion_errors += errors
            print(f"    {col}: {errors} conversion errors → set to NaN")
log_quality("T4", "silver.admissions", "Numeric type conversion",
            total_rows, conversion_errors,
            f"Converted {len(numeric_cols)} columns to numeric")

# T5: Age encoding
print("\n  [T5] Encoding age groups to numeric")
age_map = {
    "[0-10)": 5,  "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}
df_clean["age_numeric"] = df_clean["age"].map(age_map)
unmapped = df_clean["age_numeric"].isnull().sum()
log_quality("T5", "silver.admissions", "Age group mapping",
            total_rows, unmapped, f"Map applied")

# T6: Handle missing values
print("\n  [T6] Imputing missing values")
cat_fill = {
    "race": "Unknown", "payer_code": "Unknown",
    "medical_specialty": "Unknown", "gender": "Unknown"
}
for col, fill_val in cat_fill.items():
    if col in df_clean.columns:
        n_filled = df_clean[col].isnull().sum()
        df_clean[col] = df_clean[col].fillna(fill_val)
        if n_filled > 0:
            print(f"    {col}: filled {n_filled} NaN → '{fill_val}'")

num_fill_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                 "num_medications", "number_diagnoses", "age_numeric"]
for col in num_fill_cols:
    if col in df_clean.columns:
        n_filled = df_clean[col].isnull().sum()
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        if n_filled > 0:
            print(f"    {col}: filled {n_filled} NaN → median={median_val}")

log_quality("T6", "silver.admissions", "Missing value imputation",
            total_rows, df_clean.isnull().sum().sum(),
            "Cat→Unknown, Num→median")

# T7: Outlier capping
print("\n  [T7] Outlier detection and capping (IQR method)")
outlier_cols = ["time_in_hospital", "num_lab_procedures",
                "num_procedures", "num_medications", "number_diagnoses"]
total_outliers = 0
for col in outlier_cols:
    if col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_low  = (df_clean[col] < lower).sum()
        outliers_high = (df_clean[col] > upper).sum()
        n_outliers = outliers_low + outliers_high
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        total_outliers += n_outliers
        print(f"    {col:<30} low={outliers_low:>4} high={outliers_high:>4} "
              f"→ capped to [{lower:.1f}, {upper:.1f}]")
log_quality("T7", "silver.admissions", "Outlier capping (IQR)",
            total_rows, total_outliers, "Winsorization applied")

# T8: Standardize categories
print("\n  [T8] Standardizing categorical values")
gender_map = {"Male": "M", "Female": "F", "Unknown/Invalid": "U", "Unknown": "U"}
df_clean["gender"] = df_clean["gender"].map(gender_map).fillna("U")
race_map = {
    "Caucasian": "Caucasian", "AfricanAmerican": "African American",
    "Hispanic": "Hispanic", "Asian": "Asian",
    "Other": "Other", "Unknown": "Unknown"
}
df_clean["race"] = df_clean["race"].map(race_map).fillna("Unknown")
print(f"    Race values: {df_clean['race'].value_counts().to_dict()}")
print(f"    Gender values: {df_clean['gender'].value_counts().to_dict()}")

# T9: Target variable
print("\n  [T9] Creating target variable (readmitted_30)")
df_clean["readmitted_30"] = (df_clean["readmitted"] == "<30").astype(int)
target_dist = df_clean["readmitted_30"].value_counts()
print(f"    Target distribution:")
print(f"      0 (Not readmitted <30d): {target_dist.get(0,0):,} "
      f"({target_dist.get(0,0)/len(df_clean)*100:.1f}%)")
print(f"      1 (Readmitted <30d):     {target_dist.get(1,0):,} "
      f"({target_dist.get(1,0)/len(df_clean)*100:.1f}%)")
log_quality("T9", "silver.admissions", "Target variable creation",
            len(df_clean), 0,
            f"30-day readmit rate: {df_clean['readmitted_30'].mean()*100:.2f}%")

# T10: Schema alignment
print("\n  [T10] Aligning columns to Silver schema")
df_clean = df_clean.rename(columns={
    "A1Cresult": "a1c_result",
    "change": "change_flag",
    "diabetesMed": "diabetes_med"
})
silver_cols = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "age_numeric",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "diag_1", "diag_2", "diag_3",
    "insulin", "change_flag", "diabetes_med", "a1c_result",
    "readmitted", "readmitted_30"
]
available_cols = [c for c in silver_cols if c in df_clean.columns]
df_silver = df_clean[available_cols].copy()
print(f"    Silver columns: {len(df_silver.columns)}")
print(f"    Silver rows   : {len(df_silver):,}")

# ══════════════════════════════════════════════════════════════════════
# LOAD — Silver
# ══════════════════════════════════════════════════════════════════════
print("\n[LOAD] Writing to silver.admissions...")
df_silver.to_sql(
    "admissions", engine, schema="silver",
    if_exists="replace", index=False,
    chunksize=5000
)
print(f"  ✅ Silver loaded: {len(df_silver):,} rows")

# ══════════════════════════════════════════════════════════════════════
# GOLD — Dimension tables (no CASCADE needed — already dropped upfront)
# ══════════════════════════════════════════════════════════════════════
print("\n[GOLD] Loading dimension tables...")

# dim_patient
print("  Loading gold.dim_patient...")
df_patient = df_silver[["patient_nbr", "race", "gender", "age", "age_numeric"]].copy()
df_patient = df_patient.drop_duplicates(subset=["patient_nbr"])
df_patient = df_patient.rename(columns={"age": "age_group"})
df_patient.to_sql("dim_patient", engine, schema="gold",
                  if_exists="replace", index=False)
print(f"    ✅ {len(df_patient):,} unique patients loaded")

# dim_diagnosis
print("  Loading gold.dim_diagnosis...")

def get_icd_category(code):
    if pd.isna(code) or code == "Unknown":
        return "Unknown"
    code_str = str(code).upper().strip()
    try:
        num = float(code_str)
        if 1   <= num <= 139: return "Infectious & Parasitic"
        if 140 <= num <= 239: return "Neoplasms"
        if 240 <= num <= 279: return "Endocrine/Nutritional/Metabolic"
        if 280 <= num <= 289: return "Blood Disorders"
        if 290 <= num <= 319: return "Mental Disorders"
        if 320 <= num <= 389: return "Nervous System"
        if 390 <= num <= 459: return "Circulatory System"
        if 460 <= num <= 519: return "Respiratory System"
        if 520 <= num <= 579: return "Digestive System"
        if 580 <= num <= 629: return "Genitourinary System"
        if 680 <= num <= 709: return "Skin Disorders"
        if 710 <= num <= 739: return "Musculoskeletal"
        if 800 <= num <= 999: return "Injury & Poisoning"
    except ValueError:
        if code_str.startswith("V"): return "Supplementary V-codes"
        if code_str.startswith("E"): return "External Causes"
    return "Other"

all_diag_codes = pd.concat([
    df_silver["diag_1"], df_silver["diag_2"], df_silver["diag_3"]
]).dropna().unique()

df_diag = pd.DataFrame({
    "icd_code": all_diag_codes,
    "description": "ICD-9 Code " + pd.Series(all_diag_codes).astype(str),
    "category": [get_icd_category(c) for c in all_diag_codes],
    "is_chronic": False
})
df_diag = df_diag.drop_duplicates(subset=["icd_code"])
df_diag.to_sql("dim_diagnosis", engine, schema="gold",
               if_exists="replace", index=False)
print(f"    ✅ {len(df_diag):,} unique ICD codes loaded")
print(f"    Categories: {df_diag['category'].value_counts().head(5).to_dict()}")

# dim_date
print("  Loading gold.dim_date...")
date_records = []
for year in [2023, 2024, 2025]:
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            d = date(year, month, day)
            date_records.append({
                "date_key":    int(d.strftime("%Y%m%d")),
                "full_date":   d,
                "year":        d.year,
                "month":       d.month,
                "month_name":  d.strftime("%B"),
                "quarter":     (d.month - 1) // 3 + 1,
                "day_of_week": d.strftime("%A"),
                "is_weekend":  d.weekday() >= 5
            })
df_date = pd.DataFrame(date_records)
df_date.to_sql("dim_date", engine, schema="gold",
               if_exists="replace", index=False)
print(f"    ✅ {len(df_date):,} date records loaded (2023-2025)")

# fact_admission
print("  Loading gold.fact_admission...")
with engine.connect() as conn:
    patient_keys = pd.read_sql(
        "SELECT patient_nbr FROM gold.dim_patient", conn)
    diag_keys = pd.read_sql(
        "SELECT icd_code FROM gold.dim_diagnosis", conn)

patient_keys["patient_key"] = patient_keys.index + 1
diag_keys["diag_key"] = diag_keys.index + 1


df_fact = df_silver.merge(patient_keys, on="patient_nbr", how="left")
df_fact = df_fact.merge(
    diag_keys.rename(columns={"icd_code": "diag_1", "diag_key": "primary_diag_key"}),
    on="diag_1", how="left")

df_fact["insulin_flag"]           = df_fact["insulin"].isin(["Up", "Down", "Steady"])
df_fact["diabetes_med_flag"]      = df_fact["diabetes_med"] == "Yes"
df_fact["medication_change_flag"] = df_fact["change_flag"] == "Ch"
df_fact["readmitted_within_30"]   = df_fact["readmitted_30"] == 1

fact_cols = [
    "encounter_id", "patient_key", "primary_diag_key",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "insulin_flag", "diabetes_med_flag", "medication_change_flag",
    "a1c_result", "readmitted_within_30", "readmitted"
]
available_fact = [c for c in fact_cols if c in df_fact.columns]
df_fact_final = df_fact[available_fact].rename(
    columns={"readmitted": "readmitted_label"})

df_fact_final.to_sql("fact_admission", engine, schema="gold",
                     if_exists="replace", index=False, chunksize=5000)
print(f"    ✅ {len(df_fact_final):,} fact records loaded")

# ── Save processed CSVs for ML steps ──────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df_silver.to_csv("data/processed/silver_clean.csv", index=False)
df_fact_final.to_csv("data/processed/gold_features.csv", index=False)
print(f"\n✅ Saved CSV files:")
print(f"   data/processed/silver_clean.csv")
print(f"   data/processed/gold_features.csv")

# ── Audit log ──────────────────────────────────────────────────────────
print("\n[AUDIT] Writing quality log...")
if quality_issues:
    df_audit = pd.DataFrame(quality_issues)
    df_audit["run_timestamp"] = run_time
    df_audit.to_sql("quality_log", engine, schema="audit",
                    if_exists="append", index=False)
    print(f"  ✅ {len(quality_issues)} quality checks logged to audit.quality_log")

# ── Final summary ──────────────────────────────────────────────────────
end_time = datetime.now()
elapsed = (end_time - run_time).total_seconds()
print(f"\n{'='*60}")
print(f"  ETL COMPLETE")
print(f"  Duration : {elapsed:.1f} seconds")
print(f"  Records  : {len(df_silver):,} rows processed")
print(f"  Quality  : {sum(1 for q in quality_issues if q['status']=='PASS')}"
      f"/{len(quality_issues)} checks passed")
print(f"{'='*60}")