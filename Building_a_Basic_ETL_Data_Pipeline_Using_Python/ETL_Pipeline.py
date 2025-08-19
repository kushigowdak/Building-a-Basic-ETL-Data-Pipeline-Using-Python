# -------------------------------
# Stage 1: Business Analyst Task
# -------------------------------
# Key questions this pipeline enables:
# - What is the overall trend in patient satisfaction scores over time?
# - Which specialties have the highest number of doctors?
# - Which specialties contribute the most revenue?
# - Which specialties have the highest patient satisfaction?
# - What is the distribution of treatment & room costs?
# - Which specialties generate the most revenue while maintaining satisfaction?
# - Which doctors deliver the best balance of satisfaction vs cost?
# - Top 3 treatment types by feedback


# -------------------------------
# Stage 2: Data Engineer Task
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure warehouse folder exists
os.makedirs("data_warehouse", exist_ok=True)

# ---------------------------
# 1. EXTRACTION
# ---------------------------
doctor_info_df = pd.read_csv("./raw_data/doctors_info.csv")
patient_data_df = pd.read_csv("./raw_data/patients_data_with_doctor.csv")
feedback_df = pd.read_json("./raw_data/patient_feedback.json")

print("✅ Raw data loaded")

# ---------------------------
# 2. TRANSFORMATION
# ---------------------------

# Convert date columns
feedback_df["review_date"] = pd.to_datetime(feedback_df["review_date"], errors="coerce")
patient_data_df["treatment_date"] = pd.to_datetime(patient_data_df["treatment_date"], errors="coerce")

# Normalize IDs (ensure string + strip spaces)
for df in [feedback_df, patient_data_df]:
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["treatment_id"] = df["treatment_id"].astype(str).str.strip()

# Standardize treatment_id format (T### → T0###)
feedback_df["treatment_id"] = (
    feedback_df["treatment_id"]
    .str.replace("T", "", regex=False)
    .astype(int)
    .apply(lambda x: f"T{str(x).zfill(4)}")
)

# Merge feedback ↔ patient data (patient_id + treatment_id)
feedback_patient_df = pd.merge(
    feedback_df,
    patient_data_df,
    on=["patient_id"],
    how="inner"
)

# Merge with doctor info
final_df = pd.merge(
    feedback_patient_df,
    doctor_info_df,
    on="doctor_id",
    how="left"
)

# Derived fields
final_df["total_cost"] = final_df["treatment_cost"] + final_df["room_cost"]

print("Final dataframe shape:", final_df.shape)

# ---------------------------
# 3. LOAD
# ---------------------------

# Save processed CSV
processed_path = "data_warehouse/processed_patient_feedback.csv"
final_df.to_csv(processed_path, index=False)

print(f"✅ Processed data saved to {processed_path}")
