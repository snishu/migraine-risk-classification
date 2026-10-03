import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = "models"

model = joblib.load(f"{MODEL_DIR}/migraine_stacking_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
gender_encoder = joblib.load(f"{MODEL_DIR}/gender_encoder.pkl")
target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")

with open(f"{MODEL_DIR}/metrics.json") as f:
    METRICS = json.load(f)


@app.route("/")
def index():
    return render_template(
        "index.html",
        final_metrics=METRICS["final_model"],
        benchmark=METRICS["benchmark_models"],
        feature_importance=METRICS["feature_importance"],
        dataset_size=METRICS["dataset_size"],
    )


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    try:
        gender_val = gender_encoder.transform([payload.get("Gender", "Female")])[0]

        row = {
            "Age": float(payload.get("Age", 30)),
            "BMI": float(payload.get("BMI", 23)),
            "Sleep_Hours": float(payload.get("Sleep_Hours", 7)),
            "Stress_Level": float(payload.get("Stress_Level", 5)),
            "Screen_Time_Hours": float(payload.get("Screen_Time_Hours", 5)),
            "Water_Intake_Liters": float(payload.get("Water_Intake_Liters", 2)),
            "Caffeine_Intake_mg": float(payload.get("Caffeine_Intake_mg", 100)),
            "Physical_Activity_Hours": float(payload.get("Physical_Activity_Hours", 3)),
            "Skipped_Meals_Per_Week": float(payload.get("Skipped_Meals_Per_Week", 1)),
            "Family_History": int(payload.get("Family_History", 0)),
            "Hormonal_Changes": int(payload.get("Hormonal_Changes", 0)),
            "Weather_Sensitivity": int(payload.get("Weather_Sensitivity", 0)),
            "Smoking": int(payload.get("Smoking", 0)),
            "Alcohol_Consumption": int(payload.get("Alcohol_Consumption", 0)),
            "Light_Sensitivity_Score": float(payload.get("Light_Sensitivity_Score", 5)),
            "Noise_Sensitivity_Score": float(payload.get("Noise_Sensitivity_Score", 5)),
            "Prior_Migraine_Frequency_Monthly": float(payload.get("Prior_Migraine_Frequency_Monthly", 0)),
            "Gender_Enc": gender_val,
        }

        X = np.array([[row[c] for c in feature_columns]])
        X_scaled = scaler.transform(X)

        pred_idx = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        pred_label = target_encoder.classes_[pred_idx]

        prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(target_encoder.classes_, proba)}

        # Top contributing lifestyle factors (rule-based explanation layer)
        factors = []
        if row["Sleep_Hours"] < 6:
            factors.append("Low sleep duration (under 6 hrs)")
        if row["Stress_Level"] >= 7:
            factors.append("High stress level")
        if row["Water_Intake_Liters"] < 1.5:
            factors.append("Low water intake / dehydration risk")
        if row["Screen_Time_Hours"] >= 8:
            factors.append("Excessive screen time")
        if row["Family_History"] == 1:
            factors.append("Family history of migraine")
        if row["Hormonal_Changes"] == 1:
            factors.append("Hormonal changes")
        if row["Skipped_Meals_Per_Week"] >= 4:
            factors.append("Frequently skipping meals")
        if row["Caffeine_Intake_mg"] >= 300:
            factors.append("High caffeine intake")
        if row["Prior_Migraine_Frequency_Monthly"] >= 4:
            factors.append("Frequent past migraine episodes")
        if not factors:
            factors.append("No major risk triggers detected — lifestyle looks balanced")

        return jsonify({
            "success": True,
            "risk_class": pred_label,
            "probabilities": prob_dict,
            "contributing_factors": factors[:5],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)