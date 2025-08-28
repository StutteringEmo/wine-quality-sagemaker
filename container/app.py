<<<<<<< HEAD
# container/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
=======
﻿from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
from typing import List, Optional
import xgboost as xgb
import numpy as np

<<<<<<< HEAD
# The 11 feature names and order used in training
=======
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
FEATURE_ORDER = [
    "fixed acidity","volatile acidity","citric acid","residual sugar",
    "chlorides","free sulfur dioxide","total sulfur dioxide","density",
    "pH","sulphates","alcohol"
]

app = FastAPI(title="Wine Quality XGBoost", version="1.0")
<<<<<<< HEAD

# Load booster once at startup
=======
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
booster = xgb.Booster()
booster.load_model("model_artifacts/xgboost-model")

class Instance(BaseModel):
<<<<<<< HEAD
    # Accept any subset; we’ll check at runtime
=======
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
    fixed_acidity: Optional[float] = None
    volatile_acidity: Optional[float] = None
    citric_acid: Optional[float] = None
    residual_sugar: Optional[float] = None
    chlorides: Optional[float] = None
    free_sulfur_dioxide: Optional[float] = None
    total_sulfur_dioxide: Optional[float] = None
    density: Optional[float] = None
    pH: Optional[float] = None
    sulphates: Optional[float] = None
    alcohol: Optional[float] = None

@app.get("/")
def home():
    return {"ok": True, "model": "xgboost", "features": FEATURE_ORDER}

@app.post("/predict")
def predict(instances: List[Instance]):
    if not instances:
        raise HTTPException(status_code=400, detail="Empty payload")
<<<<<<< HEAD

    # Validate and build matrix in the expected feature order
=======
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
    rows = []
    for i, ins in enumerate(instances):
        row = []
        for fname in FEATURE_ORDER:
<<<<<<< HEAD
            # pydantic camel->underscore already handled by field names above
=======
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
            key = fname.replace(" ", "_")
            val = getattr(ins, key, None)
            if val is None:
                raise HTTPException(status_code=400,
<<<<<<< HEAD
                                    detail=f"Missing feature '{fname}' in instance {i}")
            row.append(float(val))
        rows.append(row)

=======
                    detail=f"Missing feature '{fname}' in instance {i}")
            row.append(float(val))
        rows.append(row)
>>>>>>> 03cf662 (Containerized FastAPI inference + GH Actions build to GHCR)
    dmat = xgb.DMatrix(np.array(rows))
    preds = booster.predict(dmat).tolist()
    return {"predictions": preds}
