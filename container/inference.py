import os
import json
import joblib
import pandas as pd

# The 11 features used for training (combined red+white) in the SAME order
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        row = [float(x) for x in request_body.strip().split(",")]
        return pd.DataFrame([row], columns=FEATURES)
    elif content_type == "application/json":
        obj = json.loads(request_body)
        arr = obj.get("features") or obj.get("data")
        return pd.DataFrame([arr], columns=FEATURES)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept="text/csv"):
    val = float(prediction[0])
    if accept == "text/csv":
        return str(val)
    elif accept == "application/json":
        return json.dumps({"prediction": val})
    else:
        raise ValueError(f"Unsupported accept: {accept}")
