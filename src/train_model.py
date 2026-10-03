import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="mako")

BASE = "/home/claude/migraine_project"
DATA_PATH = f"{BASE}/data/migraine_dataset.csv"
MODEL_DIR = f"{BASE}/models"
REPORT_DIR = f"{BASE}/reports"

CAT_COLS = ["Gender"]
TARGET = "Migraine_Risk"
CLASS_ORDER = ["Low", "Medium", "High"]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    le_gender = LabelEncoder()
    df["Gender_Enc"] = le_gender.fit_transform(df["Gender"])

    feature_cols = [c for c in df.columns if c not in CAT_COLS + [TARGET]]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    le_target = LabelEncoder()
    le_target.classes_ = np.array(CLASS_ORDER)  # fix order Low<Medium<High
    y_enc = le_target.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            X_train, X_test, scaler, le_gender, le_target, feature_cols)


def benchmark_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro"),
            "recall": recall_score(y_test, preds, average="macro"),
            "f1": f1_score(y_test, preds, average="macro"),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "train_time": time.time() - t0,
        }
        print(f"[{name}] Acc={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}  "
              f"CV-F1={results[name]['cv_f1_mean']:.4f}±{results[name]['cv_f1_std']:.4f}")

    return results


def tune_random_forest(X_train, y_train):
    print("\nTuning Random Forest (GridSearchCV)...")
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [10, 16, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"Best RF params: {grid.best_params_}  Best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def tune_gradient_boosting(X_train, y_train):
    print("\nTuning Gradient Boosting (GridSearchCV)...")
    param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.08, 0.1],
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"Best GB params: {grid.best_params_}  Best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def build_stacking_ensemble(best_rf, best_gb, X_train, y_train):
    print("\nBuilding Stacking Ensemble (RF + GB + KNN + LogReg -> Meta LogReg)...")
    estimators = [
        ("rf", best_rf),
        ("gb", best_gb),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1, passthrough=False,
    )
    stack.fit(X_train, y_train)
    return stack


def evaluate_final(model, X_test, y_test, le_target, name="Stacking Ensemble"):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
    }

    report = classification_report(y_test, preds, target_names=le_target.classes_, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== FINAL MODEL: {name} ===")
    for k, v in metrics.items():
        if k != "model_name":
            print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le_target.classes_))

    return metrics, report, cm, preds, proba


def plot_confusion_matrix(cm, classes, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Stacking Ensemble")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(results, path):
    names = list(results.keys())
    f1s = [results[n]["f1"] for n in names]
    accs = [results[n]["accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, accs, width, label="Accuracy", color="#38bdf8")
    plt.bar(x + width/2, f1s, width, label="F1 (macro)", color="#a855f7")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Benchmark Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, X_test, y_test, feature_names, path):
    result = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(8, 8))
    importances.plot(kind="barh", color="#22d3ee")
    plt.title("Permutation Feature Importance (Stacking Ensemble)")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return importances.sort_values(ascending=False)


def plot_roc_curves(y_test, proba, classes, path):
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    colors = ["#22d3ee", "#a855f7", "#f472b6"]
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - One vs Rest (Stacking Ensemble)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("MIGRAINE RISK CLASSIFICATION - TRAINING PIPELINE STARTED")
    print("=" * 70)

    (X_train, X_test, y_train, y_test, X_train_raw, X_test_raw,
     scaler, le_gender, le_target, feature_cols) = load_and_preprocess()

    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    print("\n--- BASELINE MODEL BENCHMARKING ---")
    results = benchmark_models(X_train, y_train, X_test, y_test)

    best_rf, rf_params = tune_random_forest(X_train, y_train)
    best_gb, gb_params = tune_gradient_boosting(X_train, y_train)

    stack_model = build_stacking_ensemble(best_rf, best_gb, X_train, y_train)

    metrics, report, cm, preds, proba = evaluate_final(stack_model, X_test, y_test, le_target)

    # ---- Plots ----
    plot_confusion_matrix(cm, le_target.classes_, f"{REPORT_DIR}/confusion_matrix.png")
    plot_model_comparison(results, f"{REPORT_DIR}/model_comparison.png")
    importances = plot_feature_importance(stack_model, X_test, y_test, feature_cols, f"{REPORT_DIR}/feature_importance.png")
    plot_roc_curves(y_test, proba, le_target.classes_, f"{REPORT_DIR}/roc_curves.png")

    # ---- Save artifacts ----
    joblib.dump(stack_model, f"{MODEL_DIR}/migraine_stacking_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le_gender, f"{MODEL_DIR}/gender_encoder.pkl")
    joblib.dump(le_target, f"{MODEL_DIR}/target_encoder.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns.pkl")

    benchmark_summary = {
        name: {k: v for k, v in r.items() if k != "model"} for name, r in results.items()
    }

    full_report = {
        "final_model": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_labels": list(le_target.classes_),
        "feature_columns": feature_cols,
        "feature_importance": importances.to_dict(),
        "benchmark_models": benchmark_summary,
        "best_rf_params": rf_params,
        "best_gb_params": gb_params,
        "dataset_size": {
            "total": len(X_train) + len(X_test),
            "train": len(X_train),
            "test": len(X_test),
        },
    }

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE. Artifacts saved to /models and /reports")
    print("=" * 70)


if __name__ == "__main__":
    main()
