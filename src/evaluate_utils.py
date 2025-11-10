"""
evaluate_utils.py
Utility functions for model evaluation, reporting, and visualization.
Used in train.py after model fitting to print reports, plot confusion matrix, ROC curve, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import pandas as pd
import numpy as np

# -------------------------
# Text-based evaluation
# -------------------------
def evaluate_model(name, model, X_test, y_test):
    """
    Prints classification report, ROC-AUC, and Confusion Matrix.
    Returns dict with key metrics for logging.
    """
    print(f"\n========== {name} ==========")
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = np.zeros_like(y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return {
        "model": name,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "roc_auc": auc,
    }

# -------------------------
# Visualization helpers
# -------------------------
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


def compare_models(results_list):
    """
    Takes a list of dicts returned by evaluate_model() and plots bar comparison.
    Example: results_list = [res_lr, res_rf]
    """
    df = pd.DataFrame(results_list)
    df_melt = df.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melt, x="metric", y="score", hue="model")
    plt.title("Model Comparison (Precision / Recall / F1 / ROC-AUC)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.show()
