import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pipeline_components import DateFeatureEngineer, ColumnDropper

# -----------------------------
# Config
# -----------------------------
RAW_CSV = Path("data/insurance_fraud.csv")   # change if needed
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOGISTIC_PATH = MODEL_DIR / "logistic_pipeline.pkl"
RF_PATH       = MODEL_DIR / "rf_pipeline.pkl"

TARGET = "fraud_reported"   # Y/N

ID_LEAK_COLS = [
    "policy_number", "insured_zip", "incident_location"
]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(RAW_CSV)

# map target to 0/1
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in CSV.")
df[TARGET] = df[TARGET].map({"Y": 1, "N": 0}).astype(int)

# split X/y
X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# Identify types (after we engineer dates we’ll re-select)
# -----------------------------
date_engineer = DateFeatureEngineer()
drop_ids = ColumnDropper(cols=ID_LEAK_COLS)

# We will build a small helper pipeline to engineer date features & drop IDs
prep_df = drop_ids.transform(date_engineer.transform(X))
# infer numeric vs categorical from the *engineered* frame
num_cols = prep_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = prep_df.select_dtypes(include=["object"]).columns.tolist()

# IMPORTANT: the real pipelines will repeat these steps internally to stay consistent at inference.
# -----------------------------
# ColumnTransformer definitions
# -----------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

# For Logistic Regression: OHE + StandardScaler(for numeric) — use with_mean=False to keep sparse compatibility
log_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_cols),
        ("num_scale", StandardScaler(with_mean=False), num_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# For Random Forest: OHE only (trees are scale-invariant). We pass numeric as-is too.
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_cols),
        ("num_passthrough", "passthrough", num_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# -----------------------------
# Build full pipelines
# -----------------------------
log_pipeline = Pipeline(steps=[
    ("date_features", DateFeatureEngineer()),
    ("drop_ids", ColumnDropper(cols=ID_LEAK_COLS)),
    ("preprocess", log_preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=None
    ))
])

rf_pipeline = Pipeline(steps=[
    ("date_features", DateFeatureEngineer()),
    ("drop_ids", ColumnDropper(cols=ID_LEAK_COLS)),
    ("preprocess", rf_preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------
# Fit models
# -----------------------------
log_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
def evaluate(name, pipe, X_te, y_te):
    y_pred = pipe.predict(X_te)
    # Some classifiers may not expose predict_proba; both of ours do:
    y_prob = pipe.predict_proba(X_te)[:, 1]
    print(f"\n====== {name} ======")
    print(classification_report(y_te, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_te, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("ROC-AUC not available:", e)
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))

evaluate("Logistic Regression", log_pipeline, X_test, y_test)
evaluate("Random Forest",       rf_pipeline, X_test, y_test)

# -----------------------------
# Save pipelines
# -----------------------------
joblib.dump(log_pipeline, LOGISTIC_PATH)
joblib.dump(rf_pipeline,  RF_PATH)

print(f"\nSaved:\n - {LOGISTIC_PATH}\n - {RF_PATH}")
