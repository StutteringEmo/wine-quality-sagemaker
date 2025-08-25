import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def main():
    # SageMaker I/O env
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    data_path = os.path.join(input_dir, "winequality_combined.csv")
    print(f"[INFO] Reading data from: {data_path}")

    df = pd.read_csv(data_path)  # our combined CSV is comma-separated

    drop_cols = [c for c in ["quality", "type"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["quality"]

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    model = LinearRegression()
    model.fit(X, y)

    out_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, out_path)
    print(f"[INFO] Saved model to: {out_path}")

if __name__ == "__main__":
    main()
