import sys
import platform
import ctypes
import os
import logging
import json
import traceback
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def check_system_dependencies():
    current_os = platform.system()
    if current_os == "Darwin":
        logging.info("skipping system-level dependency checks (assume OK).")
        return

    # Check libomp for XGBoost
    try:
        ctypes.CDLL("libomp.so")  # Linux
        logging.info("libomp found.")
    except OSError:
        try:
            ctypes.CDLL("libomp.dll")  # Windows
            logging.info("libomp found.")
        except OSError:
            logging.error("libomp not found! Please install OpenMP (Linux: libomp-dev, Windows: vcomp140.dll).")
            sys.exit(1)

    # Check unixODBC for pyodbc
    if current_os == "Linux":
        try:
            ctypes.CDLL("libodbc.so")  # unixODBC library
            logging.info("unixODBC found.")
        except OSError:
            logging.error("unixODBC not found! Please install unixODBC (e.g., sudo apt install unixodbc unixodbc-dev).")
            sys.exit(1)
    elif current_os == "Windows":
        logging.info("Windows detected — ensure ODBC Driver is installed.")

#  CHECK REQUIRED LIBRARIES

REQUIRED_LIBS = [
    "pyodbc",
    "pandas",
    "numpy",
    "sklearn",
    "xgboost"
]

def check_libraries():
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            logging.info(f"Library OK: {lib}")
        except ImportError:
            logging.error(f"Missing library: {lib}")
            sys.exit(1)
            

def get_db_connection():
    try:
        conn = pyodbc.connect(
            "DRIVER={PostgreSQL Unicode};"
            "SERVER=localhost;"
            "PORT=5432;"
            "DATABASE=ancillary_ml;"
            "UID=ancillary_user;"      # or your mac username
            "PWD=Password123;"         # add password if required
        )
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error("Database connection failed.")
        logging.error(str(e))
        sys.exit(1)            

# FLATTEN DATA QUERY (Last 12 Months)
# (Site + Primary SKU) → Ancillary SKU
#
# | Site | Layout | Primary SKU | Ancillary SKU | Quantity |
# | ---- | ------ | ----------- | ------------- | -------- |

FLATTEN_QUERY = """
WITH basedata AS (
    SELECT
        sps.siteid,
        sps.market,
        sps.region,
        sps.layoutid,
        sps.siteplanversion,

        p.skuvalue AS primaryskuvalue,
        p.devicename AS primarydevicename,

        a.skuvalue AS ancillaryskuvalue,
        a.devicename AS ancillarydevicename,
        a.orderquantity AS ancillaryquantity,

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

# LOAD DATA
def load_data(conn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = pd.read_sql(FLATTEN_QUERY, conn)

        if df.empty:
            logging.error("Query executed but returned 0 rows.")
            sys.exit(1)

        logging.info(f"Loaded {len(df)} rows from database.")
        return df

    except Exception as e:
        logging.error("Failed to load data from database.")
        logging.exception(e)
        sys.exit(1)


def load_from_csv(file_path: str) -> pd.DataFrame:
    REQUIRED_COLUMNS = [
        "siteid",
        "market",
        "region",
        "layoutid",
        "siteplanversion",
        "primaryskuvalue",
        "primarydevicename",
        "ancillaryskuvalue",
        "ancillarydevicename",
        "ancillaryquantity",
        "ordertype",
        "plantype_releaseversion",
        "entityid",
        "modifiedon"
    ]

    logging.info(f"Loading data from CSV: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("CSV file contains 0 rows")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    logging.info(f"CSV loaded successfully with {len(df)} rows")

    return df
        
        

# PREPARE ML DATASET (MULTI-HOT) 
def prepare_ml_dataset(df):
    import pandas as pd
    import logging

    df["label"] = 1

    pivot_df = df.pivot_table(
        index=["siteid", "market", "region", "primaryskuvalue"],
        columns="ancillaryskuvalue",
        values="label",
        fill_value=0
    )

    pivot_df.columns.name = None
    pivot_df = pivot_df.reset_index()

    # Ensure all label columns are integer and no NaNs
    feature_cols = ["siteid", "market", "region", "primaryskuvalue"]
    label_cols = [col for col in pivot_df.columns if col not in feature_cols]

    pivot_df[label_cols] = pivot_df[label_cols].fillna(0).astype(int)

    logging.info(f"ML dataset shape after pivot: {pivot_df.shape}")
    logging.info("Columns after pivot: " + str(list(pivot_df.columns)))

    return pivot_df

# TRAIN MODEL (XGBoost)
def train_model(data):
    # Encode categorical features
    le_market = LabelEncoder()
    le_region = LabelEncoder()
    le_primary = LabelEncoder()

    data["market"] = le_market.fit_transform(data["market"])
    data["region"] = le_region.fit_transform(data["region"])
    data["primaryskuvalue"] = le_primary.fit_transform(data["primaryskuvalue"])

    # Explicit feature columns
    feature_cols = ["market", "region", "primaryskuvalue"]
    
    # Label columns = all remaining after dropping feature columns and siteid
    label_cols = [col for col in data.columns if col not in feature_cols + ["siteid"]]

    X = data[feature_cols].astype(int)
    Y = data[label_cols].astype(int)  # ensure all 0/1

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
            learning_rate=0.1
        )
    )

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, Y_train)

    logging.info("XGBoost model training completed.")

    return model, X_test, Y_test, feature_cols, label_cols

    
# PRECISION@K CALCULATION
def precision_at_k(model, X_test, Y_test, label_cols, k=None):
    import numpy as np
    import logging

    probs = model.predict_proba(X_test)
    precisions = []

    # Determine K dynamically if not provided
    if k is None:
        # Option 1: average actual ancillaries per site
        k = int(np.ceil(Y_test.sum(axis=1).mean()))
        logging.info(f"Dynamic K calculated based on average actual ancillaries: {k}")

    for i in range(len(X_test)):
        row_probs = probs[i]
        sorted_indices = np.argsort(row_probs)[::-1]
        top_k_indices = sorted_indices[:k]

        actual = Y_test.iloc[i].values
        actual_set = set([label_cols[idx] for idx, val in enumerate(actual) if val == 1])
        predicted_set = set([label_cols[idx] for idx in top_k_indices])

        intersection = actual_set & predicted_set
        precision = len(intersection) / k

        precisions.append(precision)

        logging.info(f"Sample {i+1}:")
        logging.info(f"  Actual ancillaries   = {actual_set}")
        logging.info(f"  Top-{k} predicted    = {list(predicted_set)}")
        logging.info(f"  Intersection         = {intersection} → {len(intersection)}")
        logging.info(f"  Precision@{k}        = {precision:.4f}")

        # Log ancillarySku | probability table
        prob_table = sorted(
            [(label_cols[idx], float(row_probs[idx])) for idx in range(len(label_cols))],
            key=lambda x: x[1],
            reverse=True
        )
        logging.info("  Ancillary | Probability")
        for anc, score in prob_table:
            logging.info(f"    {anc:<12} | {score:.4f}")

    avg_precision_k = np.mean(precisions)
    logging.info(f"\nAverage Precision@{k}: {avg_precision_k:.4f}")
    return avg_precision_k

           
# GENERATE JSON OUTPUT (DYNAMIC K) 
def generate_json_output(model, sample_input, label_cols, k=None):
    import pandas as pd, json

    # Ensure sample_input is a DataFrame
    if isinstance(sample_input, pd.Series):
        sample_input = sample_input.to_frame().T

    # Predict probabilities
    probs = model.predict_proba(sample_input)[0]

    # Default k if not provided
    if k is None:
        k = len(label_cols)  # or some fallback

    k = int(k)

    # Build top-K list
    sku_scores = [{"ancillarySku": label_cols[i], "score": float(probs[i])}
                  for i in range(len(label_cols))]
    sku_scores.sort(key=lambda x: x["score"], reverse=True)

    return json.dumps(sku_scores[:k], indent=4)

            
# MAIN PIPELINE
# --- MAIN PIPELINE ---
def main():
    try:
        logging.info("Starting Ancillary Recommendation Pipeline...")
        #check_system_dependencies()
        check_libraries()
        
        profile = os.getenv("ENV_PROFILE", "DEV").upper()
        logging.info(f"Active Profile: {profile}")
        
        if profile == "DEV":
            df = load_from_csv("./data/ancillary_ml.csv")
                
        elif profile == "PROD":
            conn = get_db_connection()
            df = load_data(conn)
        else:
            raise ValueError(f"Invalid ENV_PROFILE: {profile}")    

        logging.info("Starting preprocessing...")
        
        if not df.empty:
            logging.info("Sample flattened data (first 5 rows):")
            logging.info("\n" + df.head(5).to_string(index=False))
        else:
            logging.warning("No data retrieved. Exiting.")
            return
        
        ml_data = prepare_ml_dataset(df)

        model, X_test, Y_test, feature_cols, label_cols = train_model(ml_data)

        # Determine dynamic K (average number of actual ancillaries per site)
        dynamic_k = int(np.ceil(Y_test.sum(axis=1).mean()))
        logging.info(f"Dynamic K for top-K prediction: {dynamic_k}")

        # Calculate Precision@K
        precision_at_k(model, X_test, Y_test, label_cols, k=dynamic_k)

        # Generate JSON for first test sample
        sample_input = X_test.iloc[0][feature_cols]  # select only feature columns!
        json_output = generate_json_output(model, sample_input, label_cols, k=dynamic_k)

        logging.info("Sample JSON Output:")
        print(json_output)

        logging.info("Pipeline completed successfully.")
        
    except Exception:
        logging.error("Pipeline failed.")
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    
    main()
    


