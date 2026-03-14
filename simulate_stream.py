# simulate_stream.py
# PURPOSE: Simulates a live hospital data feed by generating
#          one patient record every N seconds.
# RUN WITH: python simulate_stream.py

import pandas as pd
import json
import time
import random
import os
from datetime import datetime
from faker import Faker

fake = Faker('en_IN')  # Indian locale for demo

# ── Load the real dataset as a base template ───────────────────────────
# We use real column structure but replace personal identifiers with fake ones
print("Loading base dataset...")
df_real = pd.read_csv("data/raw/diabetic_data.csv", na_values=["?"])
df_real = df_real.fillna("Unknown")
print(f"Loaded {len(df_real)} base records to sample from.")

# ── List of realistic Indian hospital names ────────────────────────────
HOSPITALS = [
    "AIIMS New Delhi", "Apollo Hospitals Mumbai", "Fortis Bengaluru",
    "Medanta Gurugram", "Kokilaben Mumbai", "NIMHANS Bengaluru",
    "Christian Medical College Vellore", "SGPGI Lucknow"
]

DEPARTMENTS = [
    "Endocrinology", "Internal Medicine", "Cardiology",
    "Nephrology", "Emergency", "General Surgery"
]

# ── Function that creates one synthetic patient record ─────────────────
def generate_patient_record(record_number: int) -> dict:
    """
    Takes one random row from the real dataset,
    replaces personal info with fake data,
    adds hospital metadata, and returns as dict.
    """
    # Pick a random real row to use as clinical base
    base_row = df_real.sample(1).iloc[0].to_dict()

    # Replace identifying info with synthetic data
    base_row["patient_nbr"]     = f"PAT{random.randint(100000, 999999)}"
    base_row["encounter_id"]    = record_number + 900000
    base_row["patient_name"]    = fake.name()
    base_row["contact"]         = fake.phone_number()
    base_row["city"]            = fake.city()

    # Add hospital operational metadata
    base_row["hospital"]        = random.choice(HOSPITALS)
    base_row["department"]      = random.choice(DEPARTMENTS)
    base_row["admission_time"]  = datetime.now().isoformat()
    base_row["stream_seq"]      = record_number

    return base_row

# ── Function that writes records to a JSONL file ───────────────────────
# JSONL = one JSON object per line — easy to append and read line by line
def stream_patients(interval_seconds: int = 3, total_records: int = 200):
    """
    Generates `total_records` patient records,
    writing one every `interval_seconds` seconds.
    Output: data/stream/live_feed.jsonl
    """
    os.makedirs("data/stream", exist_ok=True)
    output_file = "data/stream/live_feed.jsonl"

    print(f"\n{'='*55}")
    print(f"  HOSPITAL LIVE FEED SIMULATOR")
    print(f"  Generating {total_records} records every {interval_seconds}s")
    print(f"  Output: {output_file}")
    print(f"{'='*55}\n")

    with open(output_file, "w") as f:  # "w" clears old data on each run
        for i in range(1, total_records + 1):
            record = generate_patient_record(i)

            # Write one line per record
            f.write(json.dumps(record) + "\n")
            f.flush()  # Immediately write to disk (important for live demo)

            # Print status to terminal
            print(
                f"[{i:03}/{total_records}] "
                f"{record['admission_time'][11:19]}  "
                f"Patient: {record['patient_name']:<20}  "
                f"Hospital: {record['hospital']:<30}  "
                f"Dept: {record['department']}"
            )

            # Wait before next record (set to 0 for fast mode)
            if interval_seconds > 0:
                time.sleep(interval_seconds)

    print(f"\n✅ Streaming complete! {total_records} records in {output_file}")
    print(f"   File size: {os.path.getsize(output_file)/1024:.1f} KB")

# ── Function to read back and verify the stream ────────────────────────
def verify_stream():
    """Read the JSONL file back and print a summary."""
    records = []
    with open("data/stream/live_feed.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    print(f"\n[Stream verification]")
    print(f"  Total records  : {len(df)}")
    print(f"  Columns        : {len(df.columns)}")
    print(f"  Hospitals seen : {df['hospital'].nunique()}")
    print(f"  Departments    : {df['department'].value_counts().to_dict()}")
    print(f"  Time range     : {df['admission_time'].min()[:19]}"
          f" → {df['admission_time'].max()[:19]}")
    return df

# ── Main entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    # For hackathon demo: use interval=0 to generate instantly
    # For live demo effect: use interval=2 to show real-time feel
    stream_patients(interval_seconds=0, total_records=500)
    verify_stream()