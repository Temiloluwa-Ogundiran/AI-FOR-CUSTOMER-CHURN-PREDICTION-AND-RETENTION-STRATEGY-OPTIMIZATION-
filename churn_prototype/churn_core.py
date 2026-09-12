"""
Churn reference prototype - core logic (Chapter 4 spec).
SYNTHETIC DEMONSTRATION DATA ONLY. Not Nigerian bank performance.
Implements: schema+validation, synthetic generator, preprocessing pipeline,
model training/calibration/selection, thresholding, explainability,
retention decision engine, persistence/audit.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score,
)
from sklearn.inspection import permutation_importance
import joblib

SEED = 42

NUMERIC_FEATURES = [
    "age", "tenure_months", "num_products", "is_active_member",
    "balance", "estimated_clv",
    "txn_avg_90d", "txn_count_30d", "app_sessions_30d", "days_since_last_txn",
    "failed_txn_rate", "num_complaints", "num_unresolved", "downtime_exposure",
    "fraud_flag", "monthly_fee",
]
CATEGORICAL_FEATURES = ["primary_channel", "region_group"]
TARGET = "churn_30d"

ALLOWED_CHANNELS = ["App", "USSD", "Internet", "Branch", "Agent"]
ALLOWED_REGIONS = ["Lagos-Island", "Lagos-Mainland", "Other"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Illustrative economics (DEMO ONLY - must be replaced by bank-approved values)
SAVE_PROB = 0.25
TREATMENT_COST = 1500.0  # NGN illustrative
CONTACT_COST = 500.0
FALSE_POSITIVE_HARM = 300.0
TP_VALUE_PER_NGN_CLV = 1.0  # TP value = CLV * SAVE_PROB


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def generate_synthetic(n=6000, seed=SEED, missing_rate=0.02):
    """Clearly-labelled synthetic schema to verify code paths. Mimics Ch.4 drivers:
    unresolved complaints, trust/fraud, failed txns, poor engagement, inactivity."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["customer_id"] = [f"SYN-{i:06d}" for i in range(n)]
    df["age"] = rng.integers(18, 70, n).astype(float)
    df["tenure_months"] = rng.integers(1, 120, n).astype(float)
    df["num_products"] = rng.choice([1, 2, 3, 4], n, p=[0.5, 0.3, 0.15, 0.05]).astype(float)
    df["is_active_member"] = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
    df["balance"] = np.round(rng.gamma(2.0, 50000, n), 2)
    df["estimated_clv"] = np.round(rng.gamma(2.5, 40000, n) + 5000, 2)
    df["txn_avg_90d"] = np.round(rng.gamma(3.0, 4.0, n), 1)
    df["txn_count_30d"] = rng.poisson(8, n).astype(float)
    df["app_sessions_30d"] = rng.poisson(10, n).astype(float)
    df["days_since_last_txn"] = rng.exponential(8, n).round(0)
    df["failed_txn_rate"] = np.clip(rng.beta(2, 12, n), 0, 1).round(3)
    df["num_complaints"] = rng.poisson(0.6, n).astype(float)
    # unresolved <= complaints
    df["num_unresolved"] = np.array([rng.integers(0, int(c) + 1) if c > 0 else 0 for c in df["num_complaints"]], dtype=float)
    df["downtime_exposure"] = rng.poisson(1.2, n).astype(float)
    df["fraud_flag"] = rng.choice([0, 1], n, p=[0.95, 0.05]).astype(float)
    df["monthly_fee"] = rng.choice([0, 50, 100, 250, 500], n, p=[0.2, 0.3, 0.25, 0.15, 0.1]).astype(float)
    df["primary_channel"] = rng.choice(ALLOWED_CHANNELS, n, p=[0.45, 0.25, 0.12, 0.1, 0.08])
    df["region_group"] = rng.choice(ALLOWED_REGIONS, n, p=[0.4, 0.4, 0.2])

    # latent churn score from thesis drivers
    z = (
        -2.6
        + 1.4 * df["num_unresolved"]
        + 0.6 * df["num_complaints"]
        + 4.0 * df["failed_txn_rate"]
        + 0.25 * df["downtime_exposure"]
        + 0.8 * df["fraud_flag"]
        + 0.06 * df["days_since_last_txn"]
        - 0.08 * df["txn_count_30d"]
        - 0.05 * df["app_sessions_30d"]
        - 0.5 * df["is_active_member"]
        - 0.2 * (df["num_products"] - 1)
        + 0.0005 * df["monthly_fee"]
    )
    p = 1 / (1 + np.exp(-z))
    # calibrate intercept so prevalence ~12.9%
    # shift to hit target prevalence approx
    p = np.clip(p * 0.55, 0.01, 0.95)
    df[TARGET] = (rng.random(n) < p).astype(int)

    # controlled missing values to test imputation path (not on target/id)
    for col in ALL_FEATURES:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan if df[col].dtype != object else None
        if df[col].dtype == object:
            df.loc[mask, col] = None
    df.attrs["synthetic"] = True
    df.attrs["note"] = "SYNTHETIC DEMONSTRATION ONLY - not Nigerian bank data"
    return df


def validate_schema(df: pd.DataFrame):
    """Type/range/null/category checks. Returns (ok, issues list)."""
    issues = []
    for c in ALL_FEATURES + [TARGET]:
        if c not in df.columns:
            issues.append(f"Missing column: {c}")
    if issues:
        return False, issues
    # bounds
    if ((df["failed_txn_rate"].dropna() < 0) | (df["failed_txn_rate"].dropna() > 1)).any():
        issues.append("failed_txn_rate out of [0,1]")
    for c in ["age", "tenure_months", "balance", "estimated_clv", "txn_avg_90d",
              "txn_count_30d", "app_sessions_30d", "days_since_last_txn",
              "num_complaints", "num_unresolved", "downtime_exposure", "monthly_fee"]:
        if (df[c].dropna() < 0).any():
            issues.append(f"{c} has negative values")
    bad_ch = set(df["primary_channel"].dropna().unique()) - set(ALLOWED_CHANNELS)
    if bad_ch:
        issues.append(f"Unknown channels (will be ignored safely at scoring): {bad_ch}")
    bad_rg = set(df["region_group"].dropna().unique()) - set(ALLOWED_REGIONS)
    if bad_rg:
        issues.append(f"Unknown regions (will be ignored safely at scoring): {bad_rg}")
    if set(df[TARGET].dropna().unique()) - {0, 1}:
        issues.append("Target must be binary 0/1")
    return (len([i for i in issues if i.startswith("Missing")]) == 0), issues


def lift_at_k(y_true, y_score, k=0.10):
    n = len(y_true)
    kk = max(1, int(n * k))
    idx = np.argsort(y_score)[::-1][:kk]
    rate_top = y_true.iloc[idx].mean() if hasattr(y_true, "iloc") else y_true[idx].mean()
    base = y_true.mean() if hasattr(y_true, "mean") else np.mean(y_true)
    return float(rate_top / base) if base > 0 else 0.0


def campaign_utility_threshold(y_true, y_score, clv, thresholds=None):
    """Illustrative utility: TP_value - contact_cost - FP harm. DEMO ONLY."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)
    best_t, best_u = 0.5, -1e18
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        contacted = (pred == 1).sum()
        # TP value uses mean CLV of TPs where available
        tp_clv = clv[(pred == 1) & (y_true == 1)].mean() if tp > 0 else 0
        tp_clv = 0 if np.isnan(tp_clv) else tp_clv
        u = tp * tp_clv * SAVE_PROB * TP_VALUE_PER_NGN_CLV - contacted * CONTACT_COST - fp * FALSE_POSITIVE_HARM
        if u > best_u:
            best_u, best_t = u, float(t)
    return best_t, float(best_u)


def evaluate(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "lift@10%": float(lift_at_k(y_true, y_score, 0.10)),
        "threshold": float(threshold),
    }


def train_and_select(df: pd.DataFrame, seed=SEED):
    """70/15/15 stratified split, train candidates, validate, select, lock test eval."""
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    clv_val = df.loc[X_val.index, "estimated_clv"].fillna(df["estimated_clv"].median())
    clv_test = df.loc[X_test.index, "estimated_clv"].fillna(df["estimated_clv"].median())

    pre = build_preprocessor()
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    logreg.fit(X_train, y_train)

    pre2 = build_preprocessor()
    base_rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, max_depth=10,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    # calibrated inside 5-fold CV on training data only
    cal_rf = Pipeline([
        ("pre", pre2),
        ("clf", CalibratedClassifierCV(base_rf, method="sigmoid", cv=5)),
    ])
    cal_rf.fit(X_train, y_train)

    # validation scores + illustrative thresholds
    sv_log = logreg.predict_proba(X_val)[:, 1]
    sv_rf = cal_rf.predict_proba(X_val)[:, 1]
    t_log, _ = campaign_utility_threshold(y_val.to_numpy(), sv_log, clv_val.to_numpy())
    t_rf, _ = campaign_utility_threshold(y_val.to_numpy(), sv_rf, clv_val.to_numpy())
    m_log = evaluate(y_val, sv_log, t_log)
    m_rf = evaluate(y_val, sv_rf, t_rf)

    # selection rule: highest validation ROC-AUC, prefer logreg within 0.02
    if m_log["roc_auc"] >= m_rf["roc_auc"] - 0.02:
        selected_name, selected_model, sel_thresh, sel_val = "logistic_regression", logreg, t_log, m_log
    else:
        selected_name, selected_model, sel_thresh, sel_val = "calibrated_random_forest", cal_rf, t_rf, m_rf

    # locked test evaluation
    st = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, st, sel_thresh)

    # permutation importance on test (global diagnostic)
    try:
        pi = permutation_importance(selected_model, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc")
        feat_names = ALL_FEATURES
        perm_df = pd.DataFrame({"feature": feat_names, "importance_mean": pi.importances_mean,
                                "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    except Exception as e:
        perm_df = pd.DataFrame({"feature": ALL_FEATURES, "importance_mean": 0, "importance_std": 0, "error": str(e)})

    report = {
        "synthetic_note": "SYNTHETIC DEMONSTRATION ONLY - software verification, not Nigerian bank performance",
        "seed": seed,
        "n_records": int(len(df)),
        "prevalence": float(y.mean()),
        "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "validation": {"logistic_regression": m_log, "calibrated_random_forest": m_rf},
        "selection_rule": "highest validation ROC-AUC; prefer logistic_regression if within 0.02",
        "selected_model": selected_name,
        "operating_threshold": float(sel_thresh),
        "test_metrics_selected": test_metrics,
        "economics_illustrative": {"save_prob": SAVE_PROB, "treatment_cost": TREATMENT_COST,
                                   "contact_cost": CONTACT_COST, "fp_harm": FALSE_POSITIVE_HARM},
        "versions": versions(),
        "allowed_channels": ALLOWED_CHANNELS,
        "allowed_regions": ALLOWED_REGIONS,
    }
    artifacts = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "logreg": logreg, "cal_rf": cal_rf, "selected": selected_model,
        "perm_df": perm_df, "report": report,
    }
    return artifacts


def versions():
    import sklearn, pandas, numpy, sys
    return {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "pandas": pandas.__version__, "numpy": numpy.__version__}


# ---- Retention decision engine ----
def risk_band(p: float):
    if p >= 0.7:
        return "Critical"
    if p >= 0.5:
        return "High"
    if p >= 0.36:
        return "Medium"
    return "Low"


def reason_codes(row: pd.Series):
    """Observable, actionable categories only. Associative, not causal."""
    reasons = []
    if row.get("num_unresolved", 0) and row["num_unresolved"] >= 1:
        reasons.append(f"Unresolved service issues ({int(row['num_unresolved'])} open)")
    elif row.get("num_complaints", 0) and row["num_complaints"] >= 2:
        reasons.append(f"Repeated complaints ({int(row['num_complaints'])})")
    if row.get("failed_txn_rate", 0) and row["failed_txn_rate"] >= 0.08:
        reasons.append(f"High failed-transaction rate ({row['failed_txn_rate']:.1%})")
    if row.get("downtime_exposure", 0) and row["downtime_exposure"] >= 3:
        reasons.append(f"Downtime exposure ({int(row['downtime_exposure'])} days)")
    if row.get("days_since_last_txn", 0) and row["days_since_last_txn"] >= 21:
        reasons.append(f"Inactivity ({int(row['days_since_last_txn'])} days since last txn)")
    elif row.get("txn_count_30d", 99) is not None and pd.notna(row.get("txn_count_30d")) and row["txn_count_30d"] <= 2:
        reasons.append("Declining engagement (<=2 txns / 30d)")
    if row.get("app_sessions_30d", 99) is not None and pd.notna(row.get("app_sessions_30d")) and row["app_sessions_30d"] <= 2:
        reasons.append("Low app engagement (<=2 sessions / 30d)")
    if row.get("num_products", 9) == 1:
        reasons.append("Single-product relationship")
    if not reasons:
        reasons.append("Elevated risk pattern (review engagement/friction signals)")
    return reasons[:3]


def recommend_action(row: pd.Series):
    """Complaint recovery > service-reliability > re-engagement > needs review."""
    if pd.notna(row.get("num_unresolved")) and row["num_unresolved"] >= 1:
        return "Complaint recovery: assign owner, resolve + follow-up call within 48h"
    if (pd.notna(row.get("failed_txn_rate")) and row["failed_txn_rate"] >= 0.08) or \
       (pd.notna(row.get("downtime_exposure")) and row["downtime_exposure"] >= 3):
        return "Assisted support: service-reliability check + fee goodwill review"
    if (pd.notna(row.get("days_since_last_txn")) and row["days_since_last_txn"] >= 21) or \
       (pd.notna(row.get("txn_count_30d")) and row["txn_count_30d"] <= 2):
        ch = row.get("primary_channel", "App")
        return f"Re-engagement via {ch}: usability nudge + low-friction check-in (suppress if opted out)"
    if row.get("num_products", 2) == 1:
        return "Needs-based review: suitability check, no automatic sale"
    return "Monitor: no contact unless risk persists next cycle"


def expected_value(p: float, clv: float):
    clv = 0 if clv is None or (isinstance(clv, float) and np.isnan(clv)) else float(clv)
    return float(p * clv * SAVE_PROB - TREATMENT_COST)


def score_customers(model, df_features: pd.DataFrame, threshold: float):
    df = df_features.copy()
    proba = model.predict_proba(df[ALL_FEATURES])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["risk_band"] = [risk_band(p) for p in proba]
    clv = df["estimated_clv"] if "estimated_clv" in df.columns else pd.Series([0] * len(df), index=df.index)
    out["expected_retention_value"] = [expected_value(p, c) for p, c in zip(proba, clv.fillna(0))]
    out["reason_codes"] = [", ".join(reason_codes(r)) for _, r in df.iterrows()]
    out["recommended_action"] = [recommend_action(r) for _, r in df.iterrows()]
    out["decision"] = np.where(proba >= threshold, "FLAG FOR REVIEW (human approval required)", "No contact")
    out["suppression_note"] = "Suppress if: opted out, complaint in escalation, contacted <14d, fraud under review"
    return out
