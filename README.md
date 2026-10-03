# 🧠 NeuroRisk — Migraine Risk Classification using Machine Learning

A final-year B.Tech (Information Technology) project that predicts a person's
**migraine risk level (Low / Medium / High)** from lifestyle and physiological
data, using a **Stacking Ensemble** of multiple ML algorithms, served through
a professional, interactive **Flask web dashboard**.

---

## 🎯 Problem Statement

Migraine is influenced by a mix of lifestyle, physiological and environmental
triggers — sleep patterns, stress, hydration, screen time, hormonal changes,
family history, and sensory sensitivities. This project builds a **3-class
classification system** that estimates a person's migraine risk category
from 18 such features, and explains *why* the model reached that conclusion.

---

## 🏗️ Project Architecture

```
migraine_project/
├── data/
│   ├── generate_dataset.py      # Synthetic, clinically-inspired dataset generator
│   └── migraine_dataset.csv     # 6,000-row dataset (18 features + target)
├── src/
│   └── train_model.py           # Full ML training pipeline
├── models/                      # Saved model, scaler, encoders, metrics.json
├── reports/                     # Generated evaluation plots (PNG)
├── static/
│   ├── css/style.css            # Dashboard styling
│   ├── js/script.js             # Live prediction + gauge/animation logic
│   └── img/                     # Copied plots for the web UI
├── templates/
│   └── index.html               # Main dashboard page
├── app.py                       # Flask application (routes + prediction API)
├── requirements.txt
└── README.md
```

## 🔬 Machine Learning Pipeline

1. **Data Preprocessing** — Label encoding for categorical fields, `StandardScaler`
   for numeric features, stratified 80/20 train-test split.
2. **Baseline Benchmarking** — 7 algorithms trained and compared using 3-fold
   Stratified Cross-Validation:
   Logistic Regression, K-Nearest Neighbors, Support Vector Machine (RBF),
   Decision Tree, Random Forest, Gradient Boosting, Naive Bayes.
3. **Hyperparameter Tuning** — `GridSearchCV` on Random Forest and Gradient
   Boosting (the two strongest baselines) to find optimal depth, estimator
   count, and learning rate.
4. **Stacking Ensemble (final model)** — Combines the tuned Random Forest,
   tuned Gradient Boosting, K-Nearest Neighbors, and Logistic Regression as
   base learners, with a Logistic Regression **meta-learner** trained on their
   out-of-fold predictions (`StackingClassifier`, `cv=3`).
5. **Evaluation** — Accuracy, macro Precision/Recall/F1, multiclass
   One-vs-Rest ROC-AUC, confusion matrix.
6. **Explainability** — Permutation Feature Importance to identify which
   lifestyle factors most influence the model's predictions, plus a
   rule-based "contributing factors" layer surfaced live in the UI.

### 📊 Final Model Performance (Stacking Ensemble)

| Metric | Score |
|---|---|
| Accuracy | ~62% |
| Precision (macro) | ~62% |
| Recall (macro) | ~62% |
| F1-score (macro) | ~62% |
| ROC-AUC (One-vs-Rest) | ~0.80 |

> **Note on scores:** The dataset is intentionally generated with realistic
> noise (like real-world health survey data) rather than a clean separable
> signal — a 3-class real-world health classification problem rarely exceeds
> 60–70% accuracy without overfitting. This keeps the project honest and
> defensible in a viva: you can explain *why* the numbers look the way they
> do, instead of presenting suspiciously perfect (and likely overfit) results.

## 🖥️ Web Application

The Flask app (`app.py`) serves a single-page dashboard with:
- An **assessment form** (sliders, toggles, dropdowns) for 18 input features
- A **live prediction API** (`/api/predict`) that returns the risk class,
  class probabilities, and top contributing factors
- An animated **risk gauge** and probability bars
- A **Model Insights** section with tabs for Model Comparison, Confusion
  Matrix, ROC Curves, and Feature Importance — all generated directly from
  the training run (`metrics.json` + PNG plots)
- A **Pipeline Architecture** diagram explaining the ML workflow

## ⚙️ Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate dataset
python data/generate_dataset.py

# 3. Train the model (creates /models and /reports artifacts)
python src/train_model.py

# 4. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 🧩 Tech Stack

- **Python**, **scikit-learn** — ML pipeline, models, evaluation
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — evaluation plots
- **Flask** — backend & REST-style prediction API
- **HTML5 / CSS3 / Vanilla JS** — interactive dashboard (no frontend framework needed)

## 🚀 Possible Extensions (for viva / future scope)

- Replace the synthetic dataset with a real, ethically-sourced clinical
  dataset (e.g. from a hospital or public health survey, with proper consent).
- Add SHAP-based explainability for per-prediction feature attribution.
- Add XGBoost/LightGBM once available, and compare against the current stack.
- Deploy the Flask app on Render/Railway/PythonAnywhere with a public link.
- Add user accounts + a history log of past assessments (with a real database).
- Convert into a mobile app front-end using the same `/api/predict` endpoint.

## ⚠️ Disclaimer

This is an academic project. It is **not a medical diagnostic tool** and
must not be used to make real health decisions. Always consult a qualified
healthcare professional for migraine diagnosis and treatment.
