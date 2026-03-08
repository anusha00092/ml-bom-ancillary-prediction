import sys
import platform
import ctypes
import os
import logging
import json
import traceback
import warnings

import pyodbc
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — SYSTEM & LIBRARY CHECKS
# ══════════════════════════════════════════════════════════════════

def check_system_dependencies():
    """
    Checks OS-level dependencies needed to run the pipeline.
    - libomp  : SKIPPED — modern XGBoost bundles its own OpenMP
    - unixODBC: checked on Linux (needed by pyodbc to talk to DB)
    """
    current_os = platform.system()
    logging.info(f"Operating system detected: {current_os}")

    # libomp check is DISABLED.
    # XGBoost pip package bundles OpenMP internally on all platforms.
    # If you ever get an OpenMP runtime crash, install manually:
    #   Linux   → sudo apt install libomp-dev
    #   Windows → install Visual C++ Redistributable
    logging.info("libomp check skipped — XGBoost manages OpenMP internally.")

    if current_os == "Linux":
        try:
            ctypes.CDLL("libodbc.so")
            logging.info("unixODBC found.")
        except OSError:
            logging.error("unixODBC not found! Run: sudo apt install unixodbc unixodbc-dev")
            sys.exit(1)

    elif current_os == "Windows":
        logging.info("Windows detected — ensure PostgreSQL ODBC Driver is installed.")

    elif current_os == "Darwin":
        logging.info("macOS detected — skipping dependency checks.")


REQUIRED_LIBS = ["pyodbc", "pandas", "numpy", "sklearn", "xgboost"]

def check_libraries():
    """Verify all required Python libraries are installed before starting."""
    all_ok = True
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            logging.info(f"  Library OK : {lib}")
        except ImportError:
            logging.error(f"  MISSING    : {lib}  →  pip install {lib}")
            all_ok = False
    if not all_ok:
        logging.error("One or more libraries are missing. Exiting.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════

def get_db_connection():
    """Connect to PostgreSQL database (PROD mode)."""
    try:
        conn = pyodbc.connect(
            "DRIVER={PostgreSQL Unicode};"
            "SERVER=localhost;"
            "PORT=5432;"
            "DATABASE=ancillary_ml;"
            "UID=ancillary_user;"
            "PWD=Password123;"
        )
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)


# SQL: fetch last 12 months of (site, primarySKU, ancillarySKU) combinations
FLATTEN_QUERY = """
WITH basedata AS (
    SELECT
        sps.siteid,
        sps.market,
        sps.region,
        sps.layoutid,
        sps.siteplanversion,
        p.skuvalue       AS primaryskuvalue,
        p.devicename     AS primarydevicename,
        a.skuvalue       AS ancillaryskuvalue,
        a.devicename     AS ancillarydevicename,
        a.orderquantity  AS ancillaryquantity,
        a.ordertype,
        a.plantype_releaseversion,
        sps.entityid,
        a.modifiedon
    FROM public.siteplanskuselections sps
    JOIN public.siteplanskuselections_selectedskufrommapping p
        ON sps.entityid = p.entityid
       AND p.skutype = 'Primary'
    JOIN public.siteplanskuselections_selectedskufrommapping a
        ON sps.entityid = a.entityid
       AND a.skutype = 'Ancillary'
       AND a.associatedprimaryskuvalue = p.skuvalue
    WHERE a.modifiedon >= CURRENT_DATE - INTERVAL '12 months'
)
SELECT * FROM basedata;
"""


def load_data(conn) -> pd.DataFrame:
    """Load data from PostgreSQL (PROD mode)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql(FLATTEN_QUERY, conn)
        if df.empty:
            logging.error("Query returned 0 rows.")
            sys.exit(1)
        logging.info(f"Loaded {len(df)} rows from database.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)


def load_from_csv(file_path: str) -> pd.DataFrame:
    """Load data from CSV file (DEV mode)."""
    REQUIRED_COLUMNS = [
        "siteid", "market", "region", "layoutid", "siteplanversion",
        "primaryskuvalue", "primarydevicename",
        "ancillaryskuvalue", "ancillarydevicename", "ancillaryquantity",
        "ordertype", "plantype_releaseversion", "entityid", "modifiedon"
    ]
    logging.info(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("CSV file is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    logging.info(f"CSV loaded: {len(df)} rows | "
                 f"{df['primaryskuvalue'].nunique()} unique primary SKUs | "
                 f"{df['ancillaryskuvalue'].nunique()} unique ancillary SKUs")
    return df


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — IDENTIFY VARIABLE ANCILLARIES
#
#  A "variable" ancillary = one that is NOT always selected with
#  a given primary SKU. Mandatory ones (rate = 1.0) are excluded.
#
#  Selection rate = times ancillary chosen with primarySKU
#                   ─────────────────────────────────────────
#                   total unique orders for that primarySKU
#
#    variable  →  0 < selection_rate < 1.0  ✅ KEPT
#    mandatory →  selection_rate = 1.0      ❌ EXCLUDED
#    never     →  selection_rate = 0.0      ❌ EXCLUDED
# ══════════════════════════════════════════════════════════════════

def identify_variable_ancillaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters raw data to keep ONLY variable ancillaries per primary SKU.
    Returns filtered DataFrame with the same columns as input.
    """
    logging.info("\nIdentifying variable ancillaries...")

    # Total unique orders per primary SKU
    total_orders = (
        df.groupby("primaryskuvalue")["entityid"]
        .nunique()
        .reset_index(name="total_orders")
    )

    # How many unique orders each (primary, ancillary) pair appears in
    pair_counts = (
        df.groupby(["primaryskuvalue", "ancillaryskuvalue"])["entityid"]
        .nunique()
        .reset_index(name="selected_count")
    )

    # Compute selection rate per pair
    pair_counts = pair_counts.merge(total_orders, on="primaryskuvalue")
    pair_counts["selection_rate"] = (
        pair_counts["selected_count"] / pair_counts["total_orders"]
    )

    # Keep only variable pairs: selected sometimes but NOT always
    variable_pairs = pair_counts[
        (pair_counts["selection_rate"] > 0.0) &
        (pair_counts["selection_rate"] < 1.0)
    ][["primaryskuvalue", "ancillaryskuvalue"]].copy()

    total_pairs     = len(pair_counts)
    mandatory_count = (pair_counts["selection_rate"] == 1.0).sum()
    variable_count  = len(variable_pairs)

    logging.info(f"  Total (primary, ancillary) pairs   : {total_pairs}")
    logging.info(f"  Mandatory (rate=1.0) — excluded    : {mandatory_count}")
    logging.info(f"  Variable  (0 < rate < 1.0) — kept  : {variable_count}")

    # Filter original DataFrame to only variable pairs
    df_variable = df.merge(variable_pairs, on=["primaryskuvalue", "ancillaryskuvalue"])
    logging.info(f"  Rows after filtering to variable   : {len(df_variable)}\n")

    return df_variable


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — BUILD MULTI-LABEL ML DATASET
#
#  Pivot filtered data into a binary matrix:
#
#  | primaryskuvalue | ANC-001 | ANC-002 | ANC-003 | ...
#  | SKU-A           |    1    |    0    |    1    | ...
#  | SKU-B           |    0    |    1    |    1    | ...
#
#  Row  = one primary SKU
#  Col  = one variable ancillary (1 = ever used, 0 = never used)
# ══════════════════════════════════════════════════════════════════

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts filtered variable ancillary data into a multi-label
    binary matrix grouped by primary SKU only (global patterns).
    """
    df = df.copy()
    df["label"] = 1

    pivot_df = df.pivot_table(
        index=["primaryskuvalue"],
        columns="ancillaryskuvalue",
        values="label",
        aggfunc="max",     # 1 if ANY order had this pair
        fill_value=0
    )

    pivot_df.columns.name = None
    pivot_df = pivot_df.reset_index()

    label_cols = [c for c in pivot_df.columns if c != "primaryskuvalue"]
    pivot_df[label_cols] = pivot_df[label_cols].fillna(0).astype(int)

    logging.info(f"ML dataset ready: "
                 f"{pivot_df.shape[0]} primary SKUs × "
                 f"{len(label_cols)} variable ancillary columns")
    return pivot_df


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAIN XGBOOST MODEL
#
#  Trains one XGBoost binary classifier per variable ancillary
#  using OneVsRestClassifier.
#
#  Feature : encoded primaryskuvalue  (integer)
#  Labels  : 0/1 for each variable ancillary column
# ══════════════════════════════════════════════════════════════════

def train_model(ml_data: pd.DataFrame):
    """
    Trains OneVsRest XGBoost classifier on the ML dataset.

    Returns:
        model        — trained classifier
        X_test       — held-out test features
        Y_test       — held-out test labels
        feature_cols — ['primaryskuvalue']
        label_cols   — list of all variable ancillary SKU names
        le_primary   — fitted LabelEncoder for primaryskuvalue
    """
    data = ml_data.copy()

    # Encode primary SKU strings → integers (XGBoost needs numeric input)
    le_primary = LabelEncoder()
    data["primaryskuvalue"] = le_primary.fit_transform(data["primaryskuvalue"])

    feature_cols = ["primaryskuvalue"]
    label_cols   = [c for c in data.columns if c not in feature_cols]

    X = data[feature_cols].astype(int)
    Y = data[label_cols].astype(int)

    # Guard for very small datasets
    if len(X) < 5:
        logging.warning("Very few primary SKUs — using all data for train and test.")
        X_train, X_test, Y_train, Y_test = X, X, Y, Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

    model = OneVsRestClassifier(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            verbosity=0
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, Y_train)

    logging.info(f"XGBoost model trained successfully.")
    logging.info(f"  Training samples          : {len(X_train)}")
    logging.info(f"  Test samples              : {len(X_test)}")
    logging.info(f"  Variable ancillary labels : {len(label_cols)}")

    return model, X_test, Y_test, feature_cols, label_cols, le_primary


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — EVALUATE  (Precision @ K)
#
#  Precision@K = how many of the top-K predicted ancillaries
#                actually appear in the real selection / K
# ══════════════════════════════════════════════════════════════════

def evaluate_precision_at_k(
    model, X_test, Y_test, label_cols: list, k: int = 10
) -> float:
    """Computes average Precision@K over all test samples."""

    probs      = model.predict_proba(X_test)
    precisions = []

    logging.info(f"\n{'='*60}")
    logging.info(f"  MODEL EVALUATION — Precision@{k}")
    logging.info(f"{'='*60}")

    for i in range(len(X_test)):
        row_probs     = probs[i]
        top_k_indices = np.argsort(row_probs)[::-1][:k]

        actual        = Y_test.iloc[i].values
        actual_set    = {label_cols[j] for j, v in enumerate(actual) if v == 1}
        predicted_set = {label_cols[j] for j in top_k_indices}
        intersection  = actual_set & predicted_set
        precision     = len(intersection) / k
        precisions.append(precision)

    avg = float(np.mean(precisions))
    logging.info(f"  Test samples       : {len(X_test)}")
    logging.info(f"  Average Precision@{k}: {avg:.4f}  "
                 f"({'good' if avg >= 0.5 else 'low — consider more data'})")
    logging.info(f"{'='*60}\n")
    return avg


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — GENERATE TOP-10 FOR ALL PRIMARY SKUs  → CSV
# ══════════════════════════════════════════════════════════════════

def generate_top10_all_skus(
    model,
    ml_data      : pd.DataFrame,
    le_primary   : LabelEncoder,
    feature_cols : list,
    label_cols   : list,
    top_k        : int = 10,
    output_path  : str = "./data/top10_variable_ancillaries.csv"
) -> pd.DataFrame:
    """
    For EVERY primary SKU → predict top-K variable ancillaries.
    Saves a CSV and returns a DataFrame.

    CSV columns: primarySku | rank | ancillarySku | score
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"  TOP-{top_k} VARIABLE ANCILLARIES — ALL PRIMARY SKUs")
    logging.info(f"{'='*60}")

    all_results = []

    for primary_sku in sorted(ml_data["primaryskuvalue"].unique()):

        if primary_sku not in le_primary.classes_:
            logging.warning(f"  Skipping '{primary_sku}' — not in LabelEncoder.")
            continue

        encoded        = le_primary.transform([primary_sku])[0]
        sample_df      = pd.DataFrame([[encoded]], columns=feature_cols)
        probs          = model.predict_proba(sample_df)[0]
        top_k_real     = min(top_k, len(label_cols))
        sorted_indices = np.argsort(probs)[::-1][:top_k_real]

        for rank, idx in enumerate(sorted_indices, 1):
            all_results.append({
                "primarySku"   : primary_sku,
                "rank"         : rank,
                "ancillarySku" : label_cols[idx],
                "score"        : round(float(probs[idx]), 4)
            })

    result_df = pd.DataFrame(all_results)

    # Save CSV
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True
    )
    result_df.to_csv(output_path, index=False)

    logging.info(f"  Primary SKUs covered : {result_df['primarySku'].nunique()}")
    logging.info(f"  Total rows           : {len(result_df)}")
    logging.info(f"  CSV saved to         : {output_path}")

    # Readable summary in logs
    logging.info(f"\n  {'Primary SKU':<28} {'Rank':<6} {'Ancillary SKU':<28} {'Score':>8}")
    logging.info(f"  {'-'*74}")
    for _, row in result_df.iterrows():
        logging.info(
            f"  {row['primarySku']:<28} {int(row['rank']):<6} "
            f"{row['ancillarySku']:<28} {row['score']:>8.4f}"
        )

    return result_df


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — PREDICT FOR A SPECIFIC PRIMARY SKU  → JSON
#
#  Output format:
#  {
#      "primarySku": "P-FIREWALL-01",
#      "top10": [
#          { "rank": 1, "ancillarySku": "ANC-005", "score": 0.9123 },
#          { "rank": 2, "ancillarySku": "ANC-012", "score": 0.8741 },
#          ...
#      ]
#  }
# ══════════════════════════════════════════════════════════════════

def predict_for_primary_sku(
    primary_sku  : str,
    model,
    ml_data      : pd.DataFrame,
    le_primary   : LabelEncoder,
    feature_cols : list,
    label_cols   : list,
    top_k        : int = 10
) -> str:
    """
    Predicts top-K variable ancillaries for ONE specific primary SKU.

    Returns: clean JSON string
    Logs   : readable ranked table
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"  PREDICTING FOR PRIMARY SKU: '{primary_sku}'")
    logging.info(f"{'='*60}")

    known_skus = list(ml_data["primaryskuvalue"].unique())

    # ── SKU not found → return helpful error JSON ─────────────────
    if primary_sku not in known_skus:
        logging.error(f"  ❌ '{primary_sku}' NOT FOUND in training data.")
        logging.info(f"  Available primary SKUs ({len(known_skus)} total):")
        for sku in sorted(known_skus):
            logging.info(f"    • {sku}")
        return json.dumps({
            "primarySku"    : primary_sku,
            "error"         : f"Primary SKU '{primary_sku}' not found in training data.",
            "availableSkus" : sorted(known_skus)
        }, indent=4)

    if primary_sku not in le_primary.classes_:
        logging.error(f"  ❌ '{primary_sku}' not in LabelEncoder. Re-train the model.")
        return json.dumps({
            "primarySku": primary_sku,
            "error": "SKU not in LabelEncoder — re-run training."
        }, indent=4)

    # ── Show historical ancillaries for context ───────────────────
    sku_row           = ml_data[ml_data["primaryskuvalue"] == primary_sku].iloc[0]
    known_ancillaries = [col for col in label_cols if sku_row.get(col, 0) == 1]
    logging.info(f"  Historical variable ancillaries in training data: {len(known_ancillaries)}")
    for i, anc in enumerate(sorted(known_ancillaries), 1):
        logging.info(f"    {i:>2}. {anc}")

    # ── Encode → predict ──────────────────────────────────────────
    encoded        = le_primary.transform([primary_sku])[0]
    sample_df      = pd.DataFrame([[encoded]], columns=feature_cols)
    probs          = model.predict_proba(sample_df)[0]
    top_k_real     = min(top_k, len(label_cols))
    sorted_indices = np.argsort(probs)[::-1][:top_k_real]

    # ── Build top-10 result ───────────────────────────────────────
    top10_list = []

    logging.info(f"\n  Top-{top_k_real} Recommended Variable Ancillaries:")
    logging.info(f"  {'Rank':<6} {'Ancillary SKU':<28} {'Score':>8}")
    logging.info(f"  {'-'*46}")

    for rank, idx in enumerate(sorted_indices, 1):
        sku_name = label_cols[idx]
        score    = round(float(probs[idx]), 4)
        logging.info(f"  {rank:<6} {sku_name:<28} {score:>8.4f}")
        top10_list.append({
            "rank"         : rank,
            "ancillarySku" : sku_name,
            "score"        : score
        })

    # ── Final JSON ────────────────────────────────────────────────
    output  = {"primarySku": primary_sku, "top10": top10_list}
    json_str = json.dumps(output, indent=4)

    logging.info(f"{'='*60}\n")
    return json_str


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def main():
    try:
        logging.info("=" * 60)
        logging.info("  ANCILLARY RECOMMENDATION PIPELINE — Starting")
        logging.info("=" * 60)

        # ── Step 1: System & library checks ──────────────────────
        check_system_dependencies()
        check_libraries()

        # ── Step 2: Load raw data ─────────────────────────────────
        profile = os.getenv("ENV_PROFILE", "DEV").upper()
        logging.info(f"\nActive profile: {profile}")

        if profile == "DEV":
            df = load_from_csv("./data/ancillary_ml.csv")
        elif profile == "PROD":
            conn = get_db_connection()
            df   = load_data(conn)
        else:
            raise ValueError(f"Invalid ENV_PROFILE '{profile}'. Use DEV or PROD.")

        logging.info(f"\nRaw data preview:\n{df.head(5).to_string(index=False)}\n")

        # ── Step 3: Filter to VARIABLE ancillaries only ──────────
        #    Removes ancillaries that are always selected (mandatory)
        df_variable = identify_variable_ancillaries(df)

        # ── Step 4: Build multi-label binary matrix ───────────────
        ml_data = prepare_ml_dataset(df_variable)

        # ── Step 5: Train XGBoost model ───────────────────────────
        model, X_test, Y_test, feature_cols, label_cols, le_primary = train_model(ml_data)

        # ── Step 6: Evaluate model quality ───────────────────────
        TOP_K = 10
        evaluate_precision_at_k(model, X_test, Y_test, label_cols, k=TOP_K)

        # ── Step 7: Generate top-10 for ALL primary SKUs ─────────
        #    Saves to CSV: ./data/top10_variable_ancillaries.csv
        generate_top10_all_skus(
            model        = model,
            ml_data      = ml_data,
            le_primary   = le_primary,
            feature_cols = feature_cols,
            label_cols   = label_cols,
            top_k        = TOP_K,
            output_path  = "./data/top10_variable_ancillaries.csv"
        )

        # ── Step 8: Predict for a SPECIFIC primary SKU ───────────
        # ✏️  Change PRIMARY_SKU_TO_CHECK to any SKU you want
        PRIMARY_SKU_TO_CHECK = "P-FIREWALL-01"

        json_result = predict_for_primary_sku(
            primary_sku  = PRIMARY_SKU_TO_CHECK,
            model        = model,
            ml_data      = ml_data,
            le_primary   = le_primary,
            feature_cols = feature_cols,
            label_cols   = label_cols,
            top_k        = TOP_K
        )

        # Print final clean JSON to console
        print("\n" + "=" * 60)
        print(f"  FINAL JSON — Top-10 Variable Ancillaries for: {PRIMARY_SKU_TO_CHECK}")
        print("=" * 60)
        print(json_result)

        logging.info("✅ Pipeline completed successfully.")

    except Exception:
        logging.error("❌ Pipeline failed.")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    main()