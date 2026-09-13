"""
KDAC-4: ML Model Training
Trains XGBoost + Random Forest ensemble to predict T20 match outcomes.
"""

import pandas as pd
import numpy as np
import os
import pickle
import sqlite3
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print(" XGBoost not installed; using GradientBoosting as fallback.")

MODEL_DIR = "data/models"
DB_PATH = "data/cricket_warehouse.db"

FEATURE_COLS = [
    "rank_diff", "win_rate_diff", "avg_score_diff", "nrr_diff",
    "ta_icc_ranking", "tb_icc_ranking",
    "ta_win_rate_last_2_years", "tb_win_rate_last_2_years",
    "ta_avg_team_score_t20", "tb_avg_team_score_t20",
    "toss_win", "temperature_c", "humidity_pct",
    "wind_speed_kmh", "dew_factor_numeric",
    "pitch_pace_rating", "pitch_spin_rating",
]

CATEGORICAL_COLS = ["toss_decision", "pitch_type", "phase"]
TARGET_COL = "team_a_won"


def load_features(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load ML feature table from SQLite warehouse."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM ml_features", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from ml_features table.")
    return df


def preprocess(df: pd.DataFrame):
    """Encode categoricals and split into X, y."""
    data = df.copy()

    # Encode categorical columns
    le_dict = {}
    for col in CATEGORICAL_COLS:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le

    # Select features
    available = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in data.columns]
    X = data[available].fillna(data[available].median())
    y = data[TARGET_COL]

    return X, y, le_dict, available


def build_models():
    """Define model candidates."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=42
        )
    return models


def train_and_evaluate(X, y):
    """Train all models and return best one + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = build_models()
    results = {}

    print("\n" + "=" * 55)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 55)

    for name, model in models.items():
        # Cross-val
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                    scoring="accuracy", n_jobs=-1)
        # Fit
        model.fit(X_train, y_train)
        # Test
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": acc,
            "auc_roc": auc,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"\n {name}")
        print(f"   CV Accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test Accuracy: {acc:.3f}")
        print(f"   AUC-ROC      : {auc:.3f}")

    # Pick best by AUC-ROC
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best = results[best_name]
    print(f"\n Best Model: {best_name} (AUC={best['auc_roc']:.3f})")
    print("\n" + classification_report(best["y_test"], best["y_pred"],
                                       target_names=["Team B Wins", "Team A Wins"]))
    return best_name, results, X_test, y_test


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    try:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": feature_names,
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False)
            return imp
    except Exception:
        pass
    return pd.DataFrame()


def save_artifacts(best_name, results, le_dict, feature_cols, model_dir: str = MODEL_DIR):
    """Persist model, encoders, and metadata."""
    os.makedirs(model_dir, exist_ok=True)
    best_model = results[best_name]["model"]

    with open(f"{model_dir}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{model_dir}/label_encoders.pkl", "wb") as f:
        pickle.dump(le_dict, f)

    meta = {
        "best_model_name": best_name,
        "feature_cols": feature_cols,
        "metrics": {
            name: {k: v for k, v in r.items() if k not in ["model", "y_test", "y_pred", "y_prob"]}
            for name, r in results.items()
        }
    }
    with open(f"{model_dir}/metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\n Model artifacts saved to: {model_dir}/")


def run_training():
    """Full training pipeline."""
    df = load_features()
    X, y, le_dict, feature_cols = preprocess(df)
    best_name, results, X_test, y_test = train_and_evaluate(X, y)
    imp = get_feature_importance(results[best_name]["model"], feature_cols)
    if not imp.empty:
        print("\n Top 10 Feature Importances:")
        print(imp.head(10).to_string(index=False))
    save_artifacts(best_name, results, le_dict, feature_cols)
    return results


if __name__ == "__main__":
    run_training()
