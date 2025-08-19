# -------------------------------
# ML Pipeline with Reverse ETL
# -------------------------------

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# ---------------------------
# 1. LOAD ENRICHED DATA
# ---------------------------
processed_path = r"C:\Users\gowda\Documents\5th sem\FDE\Building_a_Basic_ETL_Data_Pipeline_Using_Python\data_warehouse\processed_patient_feedback.csv"
df = pd.read_csv(processed_path)

# ---------------------------
# 2. FEATURE ENGINEERING
# ---------------------------

# Patient-level aggregation
agg_df = df.groupby("patient_id").agg(
    avg_satisfaction=("patient_feedback_score", "mean"),
    total_spent=("total_cost", "sum"),
    num_treatments=("treatment_id_x", "nunique"),
    last_visit=("treatment_date", "max")
).reset_index()

# Recency feature
agg_df["last_visit"] = pd.to_datetime(agg_df["last_visit"])
latest_date = agg_df["last_visit"].max()
agg_df["recency_days"] = (latest_date - agg_df["last_visit"]).dt.days

# Drop raw last_visit
agg_df = agg_df.drop(columns=["last_visit"])

# ---------------------------
# 3. SCALING + CLUSTERING
# ---------------------------
features = ["avg_satisfaction", "total_spent", "num_treatments", "recency_days"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(agg_df[features])

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
agg_df["cluster"] = kmeans.fit_predict(X_scaled)

# Assign VIP label to cluster with highest avg spending
vip_cluster = agg_df.groupby("cluster")["total_spent"].mean().idxmax()
agg_df["customer_type"] = agg_df["cluster"].apply(
    lambda x: "VIP" if x == vip_cluster else "Non-VIP"
)

# ---------------------------
# 4. MERGE BACK INTO ORIGINAL DATA
# ---------------------------
enriched_df = df.merge(agg_df[["patient_id", "customer_type"]], on="patient_id", how="left")

# ---------------------------
# 5. REVERSE ETL (EXPORT TO CSV ONLY)
# ---------------------------
import os

os.makedirs("reverse_etl", exist_ok=True)

# Export enriched dataset to CSV
csv_path = "reverse_etl/enriched_patient_feedback.csv"
enriched_df.to_csv(csv_path, index=False)

print(f"✅ Enriched data exported to {csv_path}")
