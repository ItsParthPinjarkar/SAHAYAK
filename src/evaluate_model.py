"""
Silent Zone Detection - Model Evaluation Module
Evaluates model performance, generates ROC and Precision-Recall curves,
and analyzes life-critical Recall vs Precision trade-offs.
"""

import os
import logging
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import DATASET_PATH, MODEL_SAVE_PATH, RANDOM_SEED, TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
    from data_preprocessing import clean_and_impute_raw_data
except ImportError:
    from src.config import DATASET_PATH, MODEL_SAVE_PATH, RANDOM_SEED, TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
    from src.data_preprocessing import clean_and_impute_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_plots(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Generates and saves Confusion Matrix, ROC Curve, and Precision-Recall Curve."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Normal Zone", "Silent Zone"],
        yticklabels=["Normal Zone", "Silent Zone"],
    )
    axes[0].set_title("Confusion Matrix (Disaster Risk)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="#d9534f", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate (Unnecessary Deployment)")
    axes[1].set_ylabel("True Positive Rate (Detected Silent Zones / Recall)")
    axes[1].set_title("Receiver Operating Characteristic (ROC)")
    axes[1].legend(loc="lower right")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[2].plot(recall, precision, color="#0275d8", lw=2, label="PR Curve")
    axes[2].set_xlabel("Recall (Coverage of Trapped Citizens)")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve (Life-Safety Metric)")
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_evaluation_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved evaluation plots to: {plot_path}")


def evaluate_saved_model(
    model_path: str = MODEL_SAVE_PATH,
    data_path: str = DATASET_PATH
) -> Dict[str, Any]:
    """
    Loads saved model and runs thorough evaluation against the holdout split.
    """
    if not os.path.exists(model_path):
        logger.info(f"Trained model not found at {model_path}. Auto-training model for evaluation...")
        try:
            from train_model import train_and_tune_model
        except ImportError:
            from src.train_model import train_and_tune_model
        bundle = train_and_tune_model(model_save_path=model_path)
    else:
        bundle = joblib.load(model_path)

    pipeline = bundle["pipeline"]
    model_type = bundle.get("model_type", "Unknown")

    df = pd.read_csv(data_path)
    df_clean = clean_and_impute_raw_data(df)

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COLUMN].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"       COMPREHENSIVE MODEL EVALUATION REPORT ({model_type})")
    print("=" * 60)
    print(f" Accuracy:           {acc:.4f}")
    print(f" Precision:          {prec:.4f}")
    print(f" Recall (Silent):    {rec:.4f}  <-- CRITICAL: Maximizes detection of blackout areas")
    print(f" F1-Score:           {f1:.4f}")
    print(f" ROC-AUC Score:      {auc:.4f}")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   True Normal (TN):  {cm[0, 0]}   |   False Alarm (FP):  {cm[0, 1]}")
    print(f"   Missed Silent (FN):{cm[1, 0]}   |   Detected Silent(TP):{cm[1, 1]}")
    print("-" * 60)
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=["Normal Zone", "Silent Zone"]))
    print("=" * 60)

    # Explanation of metric asymmetric priority
    print("\n[OPERATIONAL SAFETY NOTE ON RECALL PRIORITY]:")
    print("In disaster communication forecasting, an asymmetric error penalty exists:")
    print(" - FALSE POSITIVE (Type I): Model predicts a Silent Zone when communication is intact.")
    print("   Operational cost: Standby mobile emergency communication vehicle deployed unnecessarily.")
    print(" - FALSE NEGATIVE (Type II): Model predicts Normal Zone when communication is actually severed.")
    print("   Operational cost: Trapped survivors cannot contact rescue teams, leading to catastrophic loss of life.")
    print("Thus, the model is strictly configured to achieve maximum RECALL for the Silent Zone class.\n")

    output_dir = os.path.dirname(model_path)
    generate_evaluation_plots(y_test.values, y_prob, y_pred, output_dir)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    evaluate_saved_model()
