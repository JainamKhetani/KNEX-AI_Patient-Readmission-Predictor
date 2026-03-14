# verify_data.py
import pandas as pd
import os

raw_path = "data/raw/"
files = os.listdir(raw_path)
print(f"Files in data/raw/: {files}\n")

# Load the main dataset
df = pd.read_csv("data/raw/diabetic_data.csv", na_values=["?", "Unknown/Invalid"])

print("=" * 50)
print("DATASET VERIFICATION REPORT")
print("=" * 50)
print(f"\n✅ Shape             : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"✅ Memory usage      : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"✅ Duplicate rows    : {df.duplicated().sum()}")
print(f"✅ Unique patients   : {df['patient_nbr'].nunique():,}")
print(f"✅ Unique encounters : {df['encounter_id'].nunique():,}")

print("\n--- Column names ---")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2}. {col}")

print("\n--- Target variable distribution ---")
print(df["readmitted"].value_counts())
print(f"\n  30-day readmission rate: "
      f"{(df['readmitted']=='<30').mean()*100:.2f}%")

print("\n--- Missing values (columns with any missing) ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, n in missing.items():
    print(f"  {col:<30} {n:>6} missing  ({n/len(df)*100:.1f}%)")

print("\n--- Data types ---")
print(df.dtypes.value_counts())

print("\n--- Sample rows ---")
print(df.head(3).to_string())

print("\n✅ Verification complete — ready for Step 2 (ETL)")