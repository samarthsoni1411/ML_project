import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------------------------------------------------
# 1️⃣ File Reading
# --------------------------------------------------------------------
def safe_read_df(uploaded_file):
    """Read CSV or Excel file robustly with automatic encoding fallback."""
    try:
        name = getattr(uploaded_file, "name", "").lower()
        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding="latin-1")
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Upload CSV or Excel.")
    except Exception as e:
        logging.error(f"❌ Failed to read dataset: {e}")
        raise

# --------------------------------------------------------------------
# 2️⃣ Missing Value Handling
# --------------------------------------------------------------------
def drop_high_missing(df, threshold=0.4):
    """
    Drop columns with missing ratio > threshold.
    Returns (new_df, dropped_columns)
    """
    if df.empty:
        return df, []

    miss_ratio = df.isna().mean()
    dropped = miss_ratio[miss_ratio > threshold].index.tolist()
    if dropped:
        logging.info(f"Dropped columns with >{threshold*100:.0f}% missing: {', '.join(dropped)}")
    return df.drop(columns=dropped, errors="ignore"), dropped


def simple_impute(df, strategy="most_frequent"):
    """
    Smart imputation for mixed-type DataFrames.
    - Numeric columns: mean/median/most_frequent
    - Categorical columns: most_frequent only
    - Boolean columns: converted to integers (0/1)
    - Safely handles strings, NaN, ?, and missing values.
    """
    if df.empty:
        return df

    df_copy = df.copy()

    # Replace "?" or empty strings with NaN
    df_copy = df_copy.replace(["?", ""], np.nan)

    # Convert boolean columns → integers
    bool_cols = df_copy.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df_copy[bool_cols] = df_copy[bool_cols].astype(int)
        logging.info(f"Converted {len(bool_cols)} boolean column(s) to numeric for imputation.")

    # Separate numeric and categorical columns
    num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()

    try:
        # --- Numeric Imputation ---
        if num_cols:
            num_strategy = strategy if strategy in ["mean", "median", "most_frequent"] else "most_frequent"
            num_imputer = SimpleImputer(strategy=num_strategy)
            df_copy[num_cols] = num_imputer.fit_transform(df_copy[num_cols])

        # --- Categorical Imputation ---
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df_copy[cat_cols] = cat_imputer.fit_transform(df_copy[cat_cols])

        # Try numeric conversion where possible
        for col in df_copy.columns:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except Exception:
                pass

        logging.info(
            f"✅ Imputation complete | Strategy='{strategy}' | "
            f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)} | Boolean: {len(bool_cols)}"
        )
        return df_copy

    except Exception as e:
        logging.error(f"❌ Smart imputation failed: {e}")
        raise

# --------------------------------------------------------------------
# 3️⃣ Encoding Utilities
# --------------------------------------------------------------------
def encode_target_if_needed(df, target_col):
    """
    Encode target column if categorical or binary strings.
    Returns encoded array (y).
    """
    try:
        y = df[target_col].copy()
        if y.dtype == "object" or y.dtype.name == "category":
            vals = y.dropna().unique()
            if len(vals) == 2:
                mapping = {vals[0]: 0, vals[1]: 1}
                logging.info(f"Binary target encoding applied: {mapping}")
                return y.map(mapping).astype(int)
            else:
                logging.info(f"LabelEncoder applied for multi-class target '{target_col}'.")
                return LabelEncoder().fit_transform(y.fillna("MISSING"))
        else:
            return pd.Series(y).astype(float).values
    except Exception as e:
        logging.error(f"Target encoding failed: {e}")
        raise


def encode_sensitive(df, sensitive_col):
    """
    Encode a sensitive column (categorical or numeric).
    """
    try:
        s = df[sensitive_col].fillna("Unknown").copy()
        if s.dtype == "object" or s.dtype.name == "category":
            encoded = LabelEncoder().fit_transform(s)
            logging.info(f"Sensitive column '{sensitive_col}' encoded (categorical).")
            return encoded
        else:
            return pd.Series(s).astype(float).values
    except Exception as e:
        logging.error(f"Sensitive feature encoding failed: {e}")
        raise

# --------------------------------------------------------------------
# 4️⃣ Utility Helpers
# --------------------------------------------------------------------
def clean_dataframe(df, drop_thresh=0.4, impute_strategy="most_frequent"):
    """
    Full cleaning pipeline: drops high-missing, imputes, removes infinities.
    Returns cleaned DataFrame and list of dropped columns.
    """
    df, dropped = drop_high_missing(df, threshold=drop_thresh)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = simple_impute(df, strategy=impute_strategy)
    df = df.reset_index(drop=True)
    logging.info("✅ DataFrame cleaned successfully.")
    return df, dropped
