import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pickle
import os

# ---------- Adjusted dataset-specific names ----------
TARGET_COL = "Predicted Score"
DROP_COLS = ["Match ID"]
NUMERIC_COLS_DEFAULT = ["Overs Played", "Wickets Lost", "Run Rate", "Opponent Strength"]
CATEGORICAL_COLS_DEFAULT = ["Home/Away", "Pitch Condition", "Weather"]
# -----------------------------------------------------

def load_data(path):
    return pd.read_csv(path)

def quick_eda(df, show_plots=False):
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nNumeric summary:\n", df.describe().T)
    print("\nMissing values:\n", df.isnull().sum())

    if show_plots and TARGET_COL in df.columns:
        plt.figure(figsize=(8, 4))
        df[TARGET_COL].hist(bins=30)
        plt.title(f"Distribution of {TARGET_COL}")
        plt.show()

def remove_outliers_iqr(df, numeric_cols=None, k=1.5, verbose=True):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        if col not in df.columns or df[col].dropna().shape[0] < 10:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        col_mask = df[col].between(lower, upper) | df[col].isna()
        if verbose:
            print(f"Removed {(~col_mask).sum()} outliers from {col}")
        mask &= col_mask
    return df[mask].reset_index(drop=True)

def build_preprocessor(numeric_cols, categorical_cols):
    num_transformer = StandardScaler()
    try:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols)
    ], remainder="drop")

def train_and_evaluate(df, numeric_cols, categorical_cols, target_col=TARGET_COL,
                       test_size=0.2, random_state=42):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    features = [c for c in numeric_cols + categorical_cols if c in df.columns]
    X, y = df[features], df[target_col]

    # Limit rare categorical values
    for c in categorical_cols:
        if c in X.columns:
            top_vals = X[c].value_counts().nlargest(50).index
            X[c] = X[c].where(X[c].isin(top_vals), other="OTHER")

    preprocessor = build_preprocessor(
        [c for c in numeric_cols if c in X.columns],
        [c for c in categorical_cols if c in X.columns]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=random_state)
    dt_pipeline = Pipeline([("pre", preprocessor), ("model", dt)])
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    dt_r2 = r2_score(y_test, y_pred_dt)

    print(f"\nDecision Tree -> RMSE: {dt_rmse:.2f}, R2: {dt_r2:.3f}")

    # AdaBoost
    adb = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=random_state),
        n_estimators=100, random_state=random_state
    )
    adb_pipeline = Pipeline([("pre", preprocessor), ("model", adb)])
    adb_pipeline.fit(X_train, y_train)
    y_pred_adb = adb_pipeline.predict(X_test)
    adb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_adb))
    adb_r2 = r2_score(y_test, y_pred_adb)

    print(f"AdaBoost -> RMSE: {adb_rmse:.2f}, R2: {adb_r2:.3f}")

    return {
        "dt_pipeline": dt_pipeline,
        "adb_pipeline": adb_pipeline,
        "metrics": {"dt_rmse": dt_rmse, "dt_r2": dt_r2,
                    "adb_rmse": adb_rmse, "adb_r2": adb_r2},
        "X_test": X_test, "y_test": y_test,
        "y_pred_dt": y_pred_dt, "y_pred_adb": y_pred_adb,
        "numeric_cols": [c for c in numeric_cols if c in X.columns],
        "categorical_cols": [c for c in categorical_cols if c in X.columns]
    }

def predict_single(sample_dict, pipeline, numeric_cols, categorical_cols):
    Xs = pd.DataFrame({c: [sample_dict.get(c, np.nan)]
                       for c in numeric_cols + categorical_cols})
    return pipeline.predict(Xs)[0]

def save_models(results, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump(results["dt_pipeline"], open(f"{out_dir}/dt_pipeline.pkl", "wb"))
    pickle.dump(results["adb_pipeline"], open(f"{out_dir}/adb_pipeline.pkl", "wb"))

    meta = {
        "numeric_cols": results["numeric_cols"],
        "categorical_cols": results["categorical_cols"],
        "cat_values": {c: list(results["X_test"][c].dropna().unique()[:100])
                       for c in results["categorical_cols"]}
    }
    pickle.dump(meta, open(f"{out_dir}/meta_info.pkl", "wb"))
    print("\nSaved models & metadata to", os.path.abspath(out_dir))

def main_train_and_save(data_path, show_eda=False):
    print("Loading data from:", data_path)
    df = load_data(data_path)
    if show_eda:
        quick_eda(df, show_plots=False)

    if "Match ID" in df.columns:
        df = df.drop(columns=["Match ID"])

    for c in NUMERIC_COLS_DEFAULT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df_clean = remove_outliers_iqr(df, numeric_cols=NUMERIC_COLS_DEFAULT, k=1.5, verbose=True)
    results = train_and_evaluate(df_clean, NUMERIC_COLS_DEFAULT, CATEGORICAL_COLS_DEFAULT)

    save_models(results)
    print("\nMetrics:", results["metrics"])

    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(results["y_test"], results["y_pred_dt"], alpha=0.6, label="DecisionTree")
        plt.scatter(results["y_test"], results["y_pred_adb"], alpha=0.6, label="AdaBoost")
        minv, maxv = results["y_test"].min(), results["y_test"].max()
        plt.plot([minv, maxv], [minv, maxv], "k--")
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.legend()
        plt.title("True vs Predicted")
        plt.show()
    except Exception:
        pass

# ---------------- Streamlit UI ----------------
def run_streamlit_ui():
    import streamlit as st
    import io

    st.set_page_config(page_title="Cricket Score Predictor", layout="centered")
    st.title("🏏 Cricket Score Predictor (T20)")

    try:
        with open("dt_pipeline.pkl", "rb") as f:
            dt_pipeline = pickle.load(f)
        with open("adb_pipeline.pkl", "rb") as f:
            adb_pipeline = pickle.load(f)
        with open("meta_info.pkl", "rb") as f:
            meta = pickle.load(f)
    except Exception:
        st.warning("⚠️ Model files not found. Please train first with:\n\n"
                   "`python main.py --train --data t20_cricket_match_score_prediction.csv`")
        return

    numeric_cols = meta.get("numeric_cols", NUMERIC_COLS_DEFAULT)
    categorical_cols = meta.get("categorical_cols", CATEGORICAL_COLS_DEFAULT)
    cat_values = meta.get("cat_values", {})

    st.sidebar.header("📊 Match Snapshot (Inputs)")
    inputs = {}

    # Numeric inputs with sliders
    for c in numeric_cols:
        if "Overs" in c:
            inputs[c] = st.sidebar.slider(c, 0.0, 20.0, 10.0, step=0.1)
        elif "Wickets" in c:
            inputs[c] = st.sidebar.slider(c, 0, 10, 2, step=1)
        elif "Run Rate" in c:
            inputs[c] = st.sidebar.slider(c, 0.0, 20.0, 7.5, step=0.1)
        elif "Opponent Strength" in c:
            inputs[c] = st.sidebar.slider(c, 1.0, 10.0, 5.0, step=0.5)
        else:
            inputs[c] = st.sidebar.number_input(c, value=0.0)

    # Categorical dropdowns
    for c in categorical_cols:
        options = cat_values.get(c, [])
        if not options:
            if "Home/Away" in c:
                options = ["Home", "Away", "Neutral"]
            elif "Pitch" in c:
                options = ["Batting Friendly", "Bowling Friendly", "Balanced"]
            elif "Weather" in c:
                options = ["Sunny", "Cloudy", "Rainy", "Humid"]
            else:
                options = ["OTHER"]
        inputs[c] = st.sidebar.selectbox(c, options)

    if st.sidebar.button("Predict Final Score"):
        try:
            dt_pred = predict_single(inputs, dt_pipeline, numeric_cols, categorical_cols)
            adb_pred = predict_single(inputs, adb_pipeline, numeric_cols, categorical_cols)

            st.success("✅ Predictions Complete!")
            st.metric("Decision Tree", f"{dt_pred:.0f}")
            st.metric("AdaBoost", f"{adb_pred:.0f}")
            st.write("📈 Difference:", f"{adb_pred - dt_pred:.2f}")

            # ---- Download button ----
            results_df = pd.DataFrame([{
                **inputs,
                "DecisionTree_PredictedScore": round(dt_pred, 2),
                "AdaBoost_PredictedScore": round(adb_pred, 2),
                "Difference": round(adb_pred - dt_pred, 2)
            }])
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download Prediction",
                data=csv_buffer.getvalue(),
                file_name="cricket_score_prediction.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", type=str, default="t20_cricket_match_score_prediction.csv")
    parser.add_argument("--show-eda", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.train:
        main_train_and_save(args.data, show_eda=args.show_eda)
    else:
        # If run via `streamlit run main.py`, go to UI
        try:
            import streamlit as st
            run_streamlit_ui()
        except ImportError:
            print("Run with: streamlit run main.py")


#to run first in the bash type "python main.py --train --data t20_cricket_match_score_prediction.csv"

#next  in the bash try "streamlit run main.py"
print("analysis completed")