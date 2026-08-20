"""
train.py
========
Loan repayment predictor — training script.
Dataset : Lending Club  (kaggle.com/datasets/wordsforthewise/lending-club)

Uses ONLY microlearn for all ML. Pandas for CSV I/O only — zero sklearn.

Usage
-----
    python train.py                                   # synthetic demo
    python train.py data/accepted_2007_to_2018Q4.csv  # real Lending Club CSV
"""

import sys, os, pickle
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microlearn.preprocessing   import (StandardScaler, OneHotEncoder,
                                         SimpleImputer, Pipeline, ColumnTransformer)
from microlearn.tree             import RandomForestClassifier
from microlearn.model_selection  import (train_test_split, GridSearchCV,
                                         StratifiedKFold)
from microlearn                  import metrics

# ══════════════════════════════════════════════════════ feature definitions ═══

NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "dti",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    # engineered
    "loan_amnt_to_annual_inc", "installment_to_monthly_inc",
    "log_annual_inc", "years_since_earliest_cr_line",
    "years_since_last_credit_pull",
]

CATEGORICAL_COLS = [
    "term", "emp_length", "home_ownership",
    "verification_status", "purpose", "income_category",
]

ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS
NUM_IDX  = list(range(len(NUMERIC_COLS)))
CAT_IDX  = list(range(len(NUMERIC_COLS), len(ALL_COLS)))

_CUR_YR = datetime.now().year
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════ helper ════════════

def _years_ago(s):
    """'Aug-05' or 'Aug-2005' → integer years elapsed."""
    try:
        yr = int(str(s).split("-")[-1].strip())
        if yr < 100:
            yr = yr + 2000 if yr <= (_CUR_YR % 100) else yr + 1900
        return max(0, _CUR_YR - yr)
    except Exception:
        return 10

# ══════════════════════════════════════════════════════════ data loading ══════

def load_real_data(path):
    """
    Load raw Lending Club CSV (accepted_2007_to_2018Q4.csv or similar).

    Handles both:
      • Raw file  — has 'loan_status' column; derives 'repay_fail' automatically
      • Pre-processed — already has 'repay_fail' binary column
    """
    import pandas as pd

    print(f"  Loading {path} …")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Raw shape : {df.shape}")

    # ── 1. create target BEFORE dropping anything ─────────────────────────
    if "repay_fail" not in df.columns:
        if "loan_status" not in df.columns:
            raise ValueError("CSV needs either 'repay_fail' or 'loan_status' column.")
        settled = ["Fully Paid", "Charged Off", "Default"]
        df = df[df["loan_status"].isin(settled)].copy()
        df["repay_fail"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)

    # ── 2. drop identifiers and post-origination leakage ─────────────────
    drop_ids = ["Unnamed: 0", "id", "zip_code", "url", "desc",
                "emp_title", "title", "member_id"]
    drop_leakage = [
        "loan_status", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_amnt",
        "issue_d", "last_pymnt_d", "next_pymnt_d",
        "funded_amnt", "funded_amnt_inv",
        "out_prncp", "out_prncp_inv",
    ]
    df.drop(columns=drop_ids + drop_leakage, errors="ignore", inplace=True)

    # ── 3. clean percentage strings ───────────────────────────────────────
    for col, divisor in [("int_rate", 1.0), ("revol_util", 100.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                errors="coerce"
            ) / divisor

    # term: " 36 months" → "36 months"
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.strip()

    # ── 4. feature engineering ────────────────────────────────────────────
    ann_inc = (
        pd.to_numeric(df.get("annual_inc", 1), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
    )

    df["loan_amnt_to_annual_inc"]    = df["loan_amnt"] / ann_inc
    df["installment_to_monthly_inc"] = df["installment"] / (ann_inc / 12.0 + 1e-9)
    df["log_annual_inc"]             = np.log1p(ann_inc)
    df["income_category"] = (
        pd.cut(ann_inc, bins=[0, 30_000, 60_000, np.inf],
               labels=["low", "medium", "high"])
        .astype(str).replace("nan", "low")
    )

    if "earliest_cr_line" in df.columns:
        df["years_since_earliest_cr_line"] = df["earliest_cr_line"].apply(_years_ago)
        df.drop(columns=["earliest_cr_line"], inplace=True)
    else:
        df["years_since_earliest_cr_line"] = 10

    if "last_credit_pull_d" in df.columns:
        df["years_since_last_credit_pull"] = df["last_credit_pull_d"].apply(_years_ago)
        df.drop(columns=["last_credit_pull_d"], inplace=True)
    else:
        df["years_since_last_credit_pull"] = 1

    if "mths_since_last_delinq" in df.columns:
        df["mths_since_last_delinq"] = df["mths_since_last_delinq"].fillna(0)

    df.drop(columns=["annual_inc"], errors="ignore", inplace=True)

    # ── 5. fill categorical NaN ───────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # ── 6. optional sample cap (avoids multi-hour training on 2M rows) ────
    MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
    if len(df) > MAX_ROWS:
        print(f"  Sampling {MAX_ROWS:,} rows from {len(df):,} "
              f"(set MAX_ROWS in train.py to change this)")
        df = df.sample(n=MAX_ROWS, random_state=42)

    # ── 7. ensure all feature columns exist ──────────────────────────────
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = 0.0 if col in NUMERIC_COLS else "Unknown"

    # Treat any non-finite source values as missing so SimpleImputer can
    # replace them. Infinite values otherwise poison StandardScaler.
    df[NUMERIC_COLS] = df[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan)

    y = df["repay_fail"].values.astype(int)
    X = df[ALL_COLS].values.astype(object)
    return X, y


def generate_synthetic_data(n: int = 6_000):
    """Synthetic Lending Club-style data used when no CSV is provided."""
    print(f"  Generating {n} synthetic samples …")
    rng = np.random.default_rng(42)

    ann_inc     = rng.lognormal(11.0, 0.5, n).clip(12_000, 500_000)
    loan_amnt   = rng.choice([5_000,8_000,10_000,15_000,20_000,25_000,35_000],
                              n).astype(float)
    int_rate    = rng.normal(13, 4, n).clip(5, 30)
    installment = (loan_amnt / 36) * (1 + int_rate / 1_200)
    dti         = rng.normal(18, 8, n).clip(0, 40)
    delinq      = rng.poisson(0.5, n).clip(0, 10).astype(float)
    inq_6m      = rng.poisson(0.8, n).clip(0, 10).astype(float)
    mths_del    = rng.choice([0,12,24,36,48,60,72], n).astype(float)
    open_acc    = rng.integers(1, 25, n).astype(float)
    pub_rec     = rng.poisson(0.1, n).clip(0, 5).astype(float)
    revol_bal   = rng.exponential(15_000, n).clip(0, 200_000)
    revol_util  = rng.beta(2, 3, n)
    total_acc   = rng.integers(5, 50, n).astype(float)
    yrs_cr      = rng.normal(10, 5, n).clip(1, 40)
    yrs_pull    = rng.integers(0, 4, n).astype(float)

    loan_inc    = loan_amnt / ann_inc
    inst_inc    = installment / (ann_inc / 12.0 + 1e-9)
    log_inc     = np.log1p(ann_inc)
    inc_cat     = np.where(ann_inc < 30_000, "low",
                  np.where(ann_inc < 60_000, "medium", "high"))

    terms    = rng.choice(["36 months","60 months"], n, p=[0.7,0.3])
    emp_len  = rng.choice(["< 1 year","1 year","2 years","3 years","4 years",
                            "5 years","6 years","7 years","8 years",
                            "9 years","10+ years"], n)
    home_own = rng.choice(["RENT","OWN","MORTGAGE"], n, p=[0.4,0.2,0.4])
    verif    = rng.choice(["Not Verified","Verified","Source Verified"], n)
    purpose  = rng.choice(["debt_consolidation","credit_card","home_improvement",
                            "other","small_business","major_purchase"], n)

    logit = (
        # Calibrated to roughly a 20% default rate, close to the settled
        # Lending Club population.  The old -1.5 intercept generated about
        # 66% defaults and made ordinary applicants appear excessively risky.
        -4.0
        + 0.06 * int_rate
        + 0.04 * dti
        + 0.40 * delinq
        + 0.35 * (pub_rec > 0).astype(float)
        + 2.00 * loan_inc
        - 0.01 * yrs_cr
        + 0.30 * (terms == "60 months").astype(float)
        + rng.normal(0, 0.8, n)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    num_part = np.column_stack([
        loan_amnt, int_rate, installment, dti,
        delinq, inq_6m, mths_del, open_acc, pub_rec,
        revol_bal, revol_util, total_acc,
        loan_inc, inst_inc, log_inc, yrs_cr, yrs_pull,
    ])
    cat_part = np.column_stack([terms, emp_len, home_own, verif, purpose, inc_cat])
    return np.hstack([num_part, cat_part]).astype(object), y


# ═══════════════════════════════════════════════════════ pipeline factory ══════

def build_pipeline():
    """
    microlearn Pipeline
        ColumnTransformer
            num  →  SimpleImputer(mean)  →  StandardScaler
            cat  →  OneHotEncoder
        RandomForestClassifier
    No sklearn anywhere.
    """
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
        ]), NUM_IDX),
        ("cat", OneHotEncoder(), CAT_IDX),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            n_estimators=50, max_depth=10,
            min_samples_split=5, max_features="sqrt",
            random_state=42,
        )),
    ])


# ═══════════════════════════════════════════════════════════════════ main ══════

def main():
    print("=" * 64)
    print("  microlearn  ·  Loan Repayment Predictor  ·  train.py")
    print("=" * 64)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path and os.path.exists(csv_path):
        X, y = load_real_data(csv_path)
    else:
        if csv_path:
            print(f"\n  ⚠  '{csv_path}' not found — using synthetic data.")
        else:
            print("\n  No CSV supplied — using synthetic data.")
        X, y = generate_synthetic_data()

    n_total, n_def = len(y), int(y.sum())
    print(f"\n  Dataset : {n_total:,} samples  ×  {len(ALL_COLS)} features")
    print(f"  Balance : {n_total-n_def:,} repaid  /  {n_def:,} defaulted  "
          f"({100*y.mean():.1f}% default rate)\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Split   : {len(X_tr):,} train  /  {len(X_te):,} test\n")

    # ── GridSearchCV ──────────────────────────────────────────────────────
    print("─" * 54)
    print("  GridSearchCV  (4 combos × 3-fold CV)")
    print("─" * 54)
    grid = GridSearchCV(
        build_pipeline(),
        param_grid={
            "classifier__n_estimators": [30, 60],
            "classifier__max_depth":    [8, 12],
        },
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        verbose=1,
    )
    grid.fit(X_tr, y_tr)
    pipeline = grid.best_estimator_

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV acc : {grid.best_score_:.4f}")

    # ── test set evaluation ───────────────────────────────────────────────
    y_pred  = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]

    acc = metrics.accuracy_score(y_te, y_pred)
    auc = metrics.roc_auc_score(y_te, y_proba)
    cm  = metrics.confusion_matrix(y_te, y_pred)
    rpt = metrics.classification_report(y_te, y_pred,
                                         target_names=["Repaid", "Defaulted"])

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print("\n" + rpt)
    print("  Confusion matrix  (rows = true, cols = predicted):")
    for row in cm:
        print("    ", row)

    # ── feature importances ───────────────────────────────────────────────
    _, ct  = pipeline.steps[0]   # ColumnTransformer
    _, clf = pipeline.steps[1]   # RandomForestClassifier
    _, ohe, _ = ct._fitted[1]    # OneHotEncoder

    ohe_names = [
        f"{CATEGORICAL_COLS[i]}={c}"
        for i, cats in enumerate(ohe.categories_)
        for c in cats
    ]
    feat_names  = NUMERIC_COLS + ohe_names
    importances = clf.feature_importances_

    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top-10 feature importances:")
    for rank, i in enumerate(top_idx, 1):
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        bar  = "█" * int(importances[i] * 60)
        print(f"   {rank:2d}.  {name:<40}  {importances[i]:.4f}  {bar}")

    # ── save model bundle ─────────────────────────────────────────────────
    model_dir = os.path.join(PROJECT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    bundle = dict(
        pipeline     = pipeline,
        all_cols     = ALL_COLS,
        numeric_cols = NUMERIC_COLS,
        cat_cols     = CATEGORICAL_COLS,
        num_idx      = NUM_IDX,
        cat_idx      = CAT_IDX,
        feat_names   = feat_names,
        importances  = importances.tolist(),
        test_metrics = dict(
            accuracy         = float(acc),
            roc_auc          = float(auc),
            confusion_matrix = cm.tolist(),
            report           = rpt,
        ),
        training_info = dict(
            data_source = "real" if csv_path and os.path.exists(csv_path) else "synthetic",
            samples = int(n_total),
            default_rate = float(y.mean()),
        ),
    )
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print("\n  ✓  model/model.pkl saved")
    print("  ✓  run  →  python app.py\n")


if __name__ == "__main__":
    main()
