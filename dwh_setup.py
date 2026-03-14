# dwh_setup.py
# PURPOSE: Creates all database schemas and tables in PostgreSQL.
#          Run this ONCE before the ETL pipeline.
# RUN WITH: python dwh_setup.py

from sqlalchemy import create_engine, text

# ── Database connection ────────────────────────────────────────────────
# Format: postgresql://username:password@host:port/database_name
DATABASE_URL = "postgresql://admin:admin@localhost:5432/hospital_dw"
engine = create_engine(DATABASE_URL, echo=False)

# ── All SQL as one big string ──────────────────────────────────────────
DDL_SCRIPT = """

-- =====================================================================
-- BRONZE LAYER: Raw ingestion — no cleaning, exact copy of source
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS bronze;

DROP TABLE IF EXISTS bronze.raw_admissions CASCADE;
CREATE TABLE bronze.raw_admissions (
    ingest_id       SERIAL PRIMARY KEY,
    source_file     TEXT NOT NULL,
    ingested_at     TIMESTAMP DEFAULT NOW(),
    record_count    INT,
    raw_json        JSONB    -- stores the entire row as JSON
);

DROP TABLE IF EXISTS bronze.raw_stream CASCADE;
CREATE TABLE bronze.raw_stream (
    stream_id       SERIAL PRIMARY KEY,
    received_at     TIMESTAMP DEFAULT NOW(),
    hospital        TEXT,
    department      TEXT,
    raw_json        JSONB
);

-- =====================================================================
-- SILVER LAYER: Cleaned and validated data
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS silver;

DROP TABLE IF EXISTS silver.admissions CASCADE;
CREATE TABLE silver.admissions (
    encounter_id              BIGINT PRIMARY KEY,
    patient_nbr               BIGINT,
    race                      VARCHAR(50),
    gender                    VARCHAR(10),
    age                       VARCHAR(20),
    age_numeric               INT,
    admission_type_id         INT,
    discharge_disposition_id  INT,
    admission_source_id       INT,
    time_in_hospital          INT,
    num_lab_procedures        INT,
    num_procedures            INT,
    num_medications           INT,
    number_outpatient         INT,
    number_emergency          INT,
    number_inpatient          INT,
    number_diagnoses          INT,
    diag_1                    VARCHAR(20),
    diag_2                    VARCHAR(20),
    diag_3                    VARCHAR(20),
    insulin                   VARCHAR(20),
    change_flag               VARCHAR(5),
    diabetes_med              VARCHAR(5),
    a1c_result                VARCHAR(20),
    readmitted                VARCHAR(10),
    readmitted_30             SMALLINT,   -- 1 = yes, 0 = no (our TARGET)
    processed_at              TIMESTAMP DEFAULT NOW(),
    quality_flag              TEXT        -- any quality issues noted
);

-- =====================================================================
-- GOLD LAYER: Star schema for analytics (dimension + fact tables)
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS gold;

-- DIMENSION TABLE 1: Patient demographics
DROP TABLE IF EXISTS gold.dim_patient CASCADE;
CREATE TABLE gold.dim_patient (
    patient_key     SERIAL PRIMARY KEY,
    patient_nbr     BIGINT UNIQUE NOT NULL,
    race            VARCHAR(50),
    gender          VARCHAR(10),
    age_group       VARCHAR(20),
    age_numeric     INT
);

-- DIMENSION TABLE 2: Date (for time-series analysis)
DROP TABLE IF EXISTS gold.dim_date CASCADE;
CREATE TABLE gold.dim_date (
    date_key        INT PRIMARY KEY,  -- format: YYYYMMDD e.g. 20240115
    full_date       DATE,
    year            INT,
    month           INT,
    month_name      VARCHAR(15),
    quarter         INT,
    day_of_week     VARCHAR(10),
    is_weekend      BOOLEAN
);

-- DIMENSION TABLE 3: Diagnosis codes
DROP TABLE IF EXISTS gold.dim_diagnosis CASCADE;
CREATE TABLE gold.dim_diagnosis (
    diag_key        SERIAL PRIMARY KEY,
    icd_code        VARCHAR(20) UNIQUE NOT NULL,
    description     TEXT,
    category        VARCHAR(100),   -- e.g. Circulatory, Diabetes, Respiratory
    is_chronic      BOOLEAN DEFAULT FALSE
);

-- DIMENSION TABLE 4: Admission type lookup
DROP TABLE IF EXISTS gold.dim_admission_type CASCADE;
CREATE TABLE gold.dim_admission_type (
    admission_type_id   INT PRIMARY KEY,
    description         VARCHAR(100)
);
INSERT INTO gold.dim_admission_type VALUES
(1,'Emergency'),(2,'Urgent'),(3,'Elective'),
(4,'Newborn'),(5,'Not Available'),(6,'NULL'),(7,'Trauma Center'),(8,'Not Mapped');

-- DIMENSION TABLE 5: Discharge disposition
DROP TABLE IF EXISTS gold.dim_discharge CASCADE;
CREATE TABLE gold.dim_discharge (
    disposition_id  INT PRIMARY KEY,
    description     VARCHAR(200)
);
INSERT INTO gold.dim_discharge VALUES
(1,'Discharged to home'),(2,'Discharged/transferred to short term hospital'),
(3,'Discharged/transferred to SNF'),(6,'Discharged/transferred to home with home health service'),
(11,'Expired'),(13,'Hospice / home'),(18,'NULL'),(25,'Not Mapped'),(26,'Unknown');

-- FACT TABLE: One row per hospital admission (the central table)
DROP TABLE IF EXISTS gold.fact_admission CASCADE;
CREATE TABLE gold.fact_admission (
    admission_key               SERIAL PRIMARY KEY,
    encounter_id                BIGINT UNIQUE NOT NULL,

    -- Foreign keys to dimension tables
    patient_key                 INT REFERENCES gold.dim_patient(patient_key),
    admission_date_key          INT,   -- REFERENCES gold.dim_date
    primary_diag_key            INT REFERENCES gold.dim_diagnosis(diag_key),

    -- Admission details
    admission_type_id           INT REFERENCES gold.dim_admission_type(admission_type_id),
    discharge_disposition_id    INT,
    admission_source_id         INT,

    -- Clinical measurements (these are our features for ML)
    time_in_hospital            INT,
    num_lab_procedures          INT,
    num_procedures              INT,
    num_medications             INT,
    number_outpatient           INT,
    number_emergency            INT,
    number_inpatient            INT,
    number_diagnoses            INT,

    -- Medication flags
    insulin_flag                BOOLEAN,
    diabetes_med_flag           BOOLEAN,
    medication_change_flag      BOOLEAN,
    a1c_result                  VARCHAR(20),

    -- TARGET VARIABLE
    readmitted_within_30        BOOLEAN,
    readmitted_label            VARCHAR(10),

    -- Audit
    created_at                  TIMESTAMP DEFAULT NOW()
);

-- =====================================================================
-- DATA QUALITY LOG: Tracks issues found during ETL
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS audit;

DROP TABLE IF EXISTS audit.quality_log CASCADE;
CREATE TABLE audit.quality_log (
    log_id          SERIAL PRIMARY KEY,
    run_timestamp   TIMESTAMP DEFAULT NOW(),
    step_name       TEXT,
    table_name      TEXT,
    check_name      TEXT,
    records_total   INT,
    records_failed  INT,
    pass_rate_pct   NUMERIC(5,2),
    status          TEXT,   -- PASS / WARN / FAIL
    details         TEXT
);

-- =====================================================================
-- INDEXES for query performance
-- =====================================================================
CREATE INDEX IF NOT EXISTS idx_silver_patient     ON silver.admissions(patient_nbr);
CREATE INDEX IF NOT EXISTS idx_silver_readmit     ON silver.admissions(readmitted_30);
CREATE INDEX IF NOT EXISTS idx_fact_patient       ON gold.fact_admission(patient_key);
CREATE INDEX IF NOT EXISTS idx_fact_readmit       ON gold.fact_admission(readmitted_within_30);
CREATE INDEX IF NOT EXISTS idx_fact_los           ON gold.fact_admission(time_in_hospital);

"""

# ── Execute the DDL ────────────────────────────────────────────────────
print("Connecting to PostgreSQL...")
try:
    with engine.connect() as conn:
        conn.execute(text(DDL_SCRIPT))
        conn.commit()
    print("\n✅ All schemas and tables created successfully!")
    print("\nSummary of what was created:")
    print("  bronze.raw_admissions     — raw CSV data as JSONB")
    print("  bronze.raw_stream         — simulated live feed")
    print("  silver.admissions         — cleaned patient records")
    print("  gold.dim_patient          — patient demographics")
    print("  gold.dim_date             — date dimension")
    print("  gold.dim_diagnosis        — ICD code lookup")
    print("  gold.dim_admission_type   — admission type lookup (pre-filled)")
    print("  gold.dim_discharge        — discharge disposition (pre-filled)")
    print("  gold.fact_admission       — central fact table (star schema)")
    print("  audit.quality_log         — ETL quality tracking")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Make sure PostgreSQL is running and credentials are correct.")

# ── Verify by listing all tables ───────────────────────────────────────
print("\nVerifying tables exist in database...")
check_sql = """
SELECT schemaname, tablename
FROM pg_tables
WHERE schemaname IN ('bronze','silver','gold','audit')
ORDER BY schemaname, tablename;
"""
with engine.connect() as conn:
    result = conn.execute(text(check_sql))
    rows = result.fetchall()
    print(f"\n{'Schema':<12} {'Table':<30}")
    print("-" * 42)
    for row in rows:
        print(f"  {row[0]:<10} {row[1]:<30}")
    print(f"\n  Total tables: {len(rows)}")