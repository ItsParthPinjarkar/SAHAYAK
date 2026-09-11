"""
Silent Zone Detection - Model Training Pipeline
Trains, tunes, and evaluates Random Forest (and optional XGBoost) models.
Saves the production-ready inference pipeline to models/silent_zone_model.pkl.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Path setup to allow running from any CWD or as package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import (
        DATASET_PATH,
        MODEL_SAVE_PATH,
        RANDOM_SEED,
        CATEGORICAL_FEATURES,
        NUMERICAL_RAW_FEATURES,
        ENGINEERED_FEATURES,
        TARGET_COLUMN,
    )
    from data_preprocessing import validate_dataset_schema, clean_and_impute_raw_data
    from feature_engineering import DisasterFeatureEngineer
except ImportError:
    from src.config import (
        DATASET_PATH,
        MODEL_SAVE_PATH,
        RANDOM_SEED,
        CATEGORICAL_FEATURES,
        NUMERICAL_RAW_FEATURES,
        ENGINEERED_FEATURES,
        TARGET_COLUMN,
    )
    from src.data_preprocessing import validate_dataset_schema, clean_and_impute_raw_data
    from src.feature_engineering import DisasterFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Optional XGBoost support
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost is not installed in the current environment; using RandomForest as primary model.")


def build_full_pipeline(classifier) -> Pipeline:
    """
    Constructs a complete Scikit-Learn Pipeline combining:
    1. DisasterFeatureEngineer (Domain feature generator)
    2. ColumnTransformer (Scaling + One-Hot Encoding)
    3. Classifier (Random Forest or XGBoost)
    """
    all_numerical = NUMERICAL_RAW_FEATURES + ENGINEERED_FEATURES

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_numerical),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_engineer", DisasterFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return full_pipeline


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics with high-recall priority."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def train_and_tune_model(
    data_path: str = DATASET_PATH,
    model_save_path: str = MODEL_SAVE_PATH,
    use_xgboost_if_available: bool = False
) -> Dict[str, Any]:
    """
    Complete end-to-end training routine:
    1. Loads dataset
    2. Validates schema
    3. Splits X and y with stratification
    4. Tunes hyperparameters with GridSearchCV (scoring='recall' to protect human lives)
    5. Evaluates model performance
    6. Saves production pipeline to disk
    """
    logger.info("=== STARTING SILENT ZONE MODEL TRAINING PIPELINE ===")

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset at {data_path}")

    df_raw = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df_raw)} records from {data_path}")

    # 2. Validate
    is_valid, errors = validate_dataset_schema(df_raw)
    if not is_valid:
        logger.warning(f"Data validation issues encountered: {errors}")

    df_cleaned = clean_and_impute_raw_data(df_raw)

    # 3. Separate X and y
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
    X = df_cleaned[feature_cols].copy()
    y = df_cleaned[TARGET_COLUMN].astype(int)

    logger.info(f"Target distribution: Normal (0) = {(y == 0).sum()}, Silent Zone (1) = {(y == 1).sum()}")

    # 4. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 5. Define Model & Hyperparameter Grid
    # Prioritizing Recall: Missing a true Silent Zone leaves trapped citizens stranded without communication
    if use_xgboost_if_available and XGBOOST_AVAILABLE:
        logger.info("Initializing XGBoost Classifier with scale_pos_weight...")
        scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        base_clf = XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            scale_pos_weight=scale_pos
        )
        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        }
        model_name = "XGBoost"
    else:
        logger.info("Initializing Random Forest Classifier with balanced class weights...")
        base_clf = RandomForestClassifier(
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [4, 6, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        }
        model_name = "RandomForest"

    pipeline = build_full_pipeline(base_clf)

    # 6. Hyperparameter Optimization targeting RECALL
    logger.info(f"Performing GridSearchCV for {model_name} targeting 'recall' metric...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",  # Critical for disaster safety
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    logger.info(f"Best Hyperparameters: {grid_search.best_params_}")
    logger.info(f"Best Cross-Validation Recall: {grid_search.best_score_:.4f}")

    # 7. Evaluate on Unseen Hold-out Test Set
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

    logger.info("\n" + "=" * 50)
    logger.info("TEST SET EVALUATION RESULTS:")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}  <-- Disaster Critical Metric")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"Confusion Matrix (TN, FP / FN, TP):\n{np.array(metrics['confusion_matrix'])}")
    logger.info("=" * 50)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Normal Zone (0)", "Silent Zone (1)"]))

    # 8. Save Model Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bundle = {
        "pipeline": best_pipeline,
        "model_type": model_name,
        "metrics": metrics,
        "best_params": grid_search.best_params_,
        "feature_names": feature_cols,
    }
    joblib.dump(bundle, model_save_path)
    logger.info(f"Production model bundle successfully saved to: {model_save_path}")

    return bundle


if __name__ == "__main__":
    train_and_tune_model()
