# -------------------------------
#  Data Analyst Task
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

processed_path = r"C:\Users\gowda\Documents\5th sem\FDE\Building_a_Basic_ETL_Data_Pipeline_Using_Python\data_warehouse\processed_patient_feedback.csv"

# 1. Access processed data
df = pd.read_csv(processed_path)
print("DataFrame columns:", df.columns)

# Extract month from review_date for trend analysis
df["review_date"] = pd.to_datetime(df["review_date"], errors='coerce')
df["month"] = df["review_date"].dt.to_period("M").astype(str)

#1. Overall trend in patient feedback score over time
trend = df.groupby("month", as_index=False).agg(
    avg_feedback_score=("patient_feedback_score", "mean"),
    count=("patient_feedback_score","size")
)
plt.plot(trend["month"], trend["avg_feedback_score"])
plt.title("Average Patient Feedback Score Over Time")
plt.xlabel("Month"); plt.ylabel("Avg Feedback Score"); plt.grid(True)
plt.show()

#2. Specialties with the highest number of doctors
spec_doctors = df.groupby("specialty")["doctor_id"].nunique().reset_index(name="n_doctors")
print(spec_doctors.sort_values("n_doctors", ascending=False).head())

#3. Specialties that contribute the most revenue (using total_cost as revenue)
spec_revenue = df.groupby("specialty").agg(
    total_revenue=("total_cost","sum"),
    avg_revenue=("total_cost","mean")
).reset_index().sort_values("total_revenue", ascending=False)
print(spec_revenue.head())

#4. Specialties with the highest patient feedback score
spec_feedback = df.groupby("specialty")["patient_feedback_score"].mean().reset_index()
print(spec_feedback.sort_values("patient_feedback_score", ascending=False).head())

#5. Distribution of treatment & room costs
plt.hist(df["treatment_cost"].dropna(), bins=30)
plt.title("Treatment Cost Distribution"); plt.show()

plt.hist(df["room_cost"].dropna(), bins=30)
plt.title("Room Cost Distribution"); plt.show()
#6. Specialties generating most revenue while maintaining high feedback
median_feedback = df["patient_feedback_score"].median()
rev_feedback = df.groupby("specialty").agg(
    total_revenue=("total_cost","sum"),
    avg_feedback_score=("patient_feedback_score","mean")
).reset_index()
rev_feedback_filtered = rev_feedback[rev_feedback["avg_feedback_score"] >= median_feedback]
print(rev_feedback_filtered.sort_values("total_revenue", ascending=False).head())

#7. Doctors delivering best balance of feedback vs cost
doctor_perf = df.groupby(["doctor_id","doctor_name"]).agg(
    avg_feedback_score=("patient_feedback_score","mean"),
    avg_cost=("total_cost","mean")
).reset_index()
# Simple score = feedback / cost (lower cost is better)
doctor_perf["value_score"] = doctor_perf["avg_feedback_score"] / doctor_perf["avg_cost"]
print(doctor_perf.sort_values("value_score", ascending=False).head())

# 8. Top 3 treatment types by feedback
treat_feedback = df.groupby("treatment_type").agg(
    avg_feedback=("patient_feedback_score","mean"),
    n=("patient_feedback_score","size")
).reset_index()

top3_treat = treat_feedback.sort_values(
    ["avg_feedback","n"], ascending=False
).head(3)

print(top3_treat)

# Plot Top 3 treatments
plt.bar(top3_treat["treatment_type"], top3_treat["avg_feedback"])
plt.title("Top 3 Treatment Types by Feedback")
plt.xlabel("Treatment Type")
plt.ylabel("Average Feedback")
plt.ylim(0, top3_treat["avg_feedback"].max() + 1)
plt.show()


# 9. Top 3 doctors by feedback vs cost
top3_doctors = doctor_perf.sort_values("value_score", ascending=False).head(3)
print(top3_doctors)

# Plot Top 3 doctors
plt.bar(top3_doctors["doctor_name"], top3_doctors["value_score"])
plt.title("Top 3 Doctors by Feedback vs Cost")
plt.xlabel("Doctor")
plt.ylabel("Value Score (Higher = Better)")
plt.ylim(0, top3_doctors["value_score"].max() + 1)
plt.show()
