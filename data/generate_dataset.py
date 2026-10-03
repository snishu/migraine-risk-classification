"""
Migraine Risk Dataset Generator
--------------------------------
Generates a realistic, clinically-inspired synthetic dataset for
Migraine Risk Classification (Low / Medium / High).

The feature-to-risk relationships are modeled loosely on known
migraine trigger research (sleep deprivation, stress, dehydration,
screen time, hormonal changes, family history, skipped meals,
weather/light/noise sensitivity, caffeine & smoking habits) so that
the resulting dataset has genuine, learnable signal plus realistic
noise -- exactly like real-world health survey data.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 6000

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n=N_SAMPLES):
    age = rng.integers(15, 70, n)
    gender = rng.choice(["Female", "Male", "Other"], size=n, p=[0.55, 0.42, 0.03])
    bmi = np.clip(rng.normal(24.5, 4.5, n), 15, 45)

    sleep_hours = np.clip(rng.normal(6.5, 1.6, n), 2, 11)
    stress_level = np.clip(rng.normal(5.2, 2.2, n), 1, 10)
    screen_time = np.clip(rng.normal(6.0, 2.8, n), 0, 16)
    water_intake = np.clip(rng.normal(2.1, 0.8, n), 0.3, 5)
    caffeine_intake = np.clip(rng.normal(150, 110, n), 0, 600)
    physical_activity = np.clip(rng.normal(3.0, 2.2, n), 0, 14)
    skipped_meals = np.clip(rng.poisson(2.0, n), 0, 14)

    family_history = rng.choice([0, 1], size=n, p=[0.65, 0.35])
    hormonal_changes = np.where(
        gender == "Female", rng.choice([0, 1], size=n, p=[0.55, 0.45]), 0
    )
    weather_sensitivity = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    smoking = rng.choice([0, 1], size=n, p=[0.82, 0.18])
    alcohol = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    light_sensitivity = np.clip(rng.normal(5, 2.5, n), 1, 10)
    noise_sensitivity = np.clip(rng.normal(4.6, 2.4, n), 1, 10)
    prior_migraine_freq = np.clip(rng.poisson(2.0, n), 0, 20)  # per month, history

    # ---- Latent risk score (weighted, clinically-inspired) ----
    score = (
        0.9 * (7.5 - sleep_hours).clip(min=0)          # sleep deprivation
        + 0.85 * stress_level
        + 0.35 * screen_time
        + 0.7 * (2.5 - water_intake).clip(min=0) * 2    # dehydration
        + 0.55 * (caffeine_intake / 100).clip(max=5)
        - 0.25 * physical_activity                       # exercise protective
        + 0.6 * skipped_meals
        + 2.6 * family_history
        + 2.2 * hormonal_changes
        + 1.4 * weather_sensitivity
        + 1.1 * smoking
        + 0.6 * alcohol
        + 0.5 * light_sensitivity
        + 0.45 * noise_sensitivity
        + 0.75 * prior_migraine_freq
        + 0.08 * (bmi - 22).clip(min=0)
        + rng.normal(0, 3.2, n)                          # real-world noise
    )

    # bin into 3 classes using quantile-based dynamic thresholds
    low_th, high_th = np.quantile(score, [0.40, 0.75])
    risk = np.where(score <= low_th, "Low",
            np.where(score <= high_th, "Medium", "High"))

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "BMI": np.round(bmi, 1),
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Stress_Level": np.round(stress_level, 1),
        "Screen_Time_Hours": np.round(screen_time, 1),
        "Water_Intake_Liters": np.round(water_intake, 2),
        "Caffeine_Intake_mg": np.round(caffeine_intake, 0),
        "Physical_Activity_Hours": np.round(physical_activity, 1),
        "Skipped_Meals_Per_Week": skipped_meals,
        "Family_History": family_history,
        "Hormonal_Changes": hormonal_changes,
        "Weather_Sensitivity": weather_sensitivity,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol,
        "Light_Sensitivity_Score": np.round(light_sensitivity, 1),
        "Noise_Sensitivity_Score": np.round(noise_sensitivity, 1),
        "Prior_Migraine_Frequency_Monthly": prior_migraine_freq,
        "Migraine_Risk": risk,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/migraine_project/data/migraine_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["Migraine_Risk"].value_counts())
    print(df.head())
