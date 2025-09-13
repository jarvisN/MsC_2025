# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
  title="Iris Classification API",
  version="1.0.0",
  description="Expose both baseline and augmented Iris models."
)

# ────────────────────────────────────────────────────────────────────────────
# Enable CORS so browsers can call us from any origin (e.g. file:// or your domain)
# ────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # * is to permit any access from anywhere
    allow_credentials=True,
    allow_methods=["*"],            # GET, POST, etc.
    allow_headers=["*"],            # Authorization, Content-Type, etc.
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load both models at startup
# ──────────────────────────────────────────────────────────────────────────────
try:                                # to test the model if it runs
  baseline_model  = joblib.load("models/iris-baseline.pkl")
  augmented_model = joblib.load("models/iris-augmented.pkl")
except Exception as e:
  raise RuntimeError(f"Failed to load models: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Request schema ### what inputs should look like
# ──────────────────────────────────────────────────────────────────────────────
class InputData(BaseModel):
  sepal_length: float = Field(
    ...,              # “Ellipsis” means this field is required (no default)
    gt=0,             # “greater than 0” — value must be strictly positive
    description="Sepal length in cm"
  )
  sepal_width:  float = Field(..., gt=0, description="Sepal width in cm")
  petal_length: float = Field(..., gt=0, description="Petal length in cm")
  petal_width:  float = Field(..., gt=0, description="Petal width in cm")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Health‐check endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", summary="API health check") ### .get is a http method which means other http method can be put here
def health_check():
  return {
    "status": "ok",
    "baseline_loaded":  True,
    "augmented_loaded": True
  }

@app.get("/myname", summary="Display my name") ### added line for another API line
def myname():
  return {
      #Obj - Json     ### learn more about this later
      #Key: Value
      "name": "HD",
      "age": 29,
  }

# ──────────────────────────────────────────────────────────────────────────────
# 4. Shared prediction logic
# ──────────────────────────────────────────────────────────────────────────────

def do_predict(model, data: InputData):
  # Define expected feature names
  expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

  # output class name
  iris_target_names = ['setosa', 'versicolor', 'virginica']
    
  # Wrap input in a DataFrame with correct feature names
  input_df = pd.DataFrame([[
      data.sepal_length,
      data.sepal_width,
      data.petal_length,
      data.petal_width
  ]], columns=expected_columns)

  # Get prediction index (e.g., 0, 1, 2)
  pred_idx = model.predict(input_df)[0]

  # Map to class name iris_target_names (manually hard-code)
  pred_name = iris_target_names[pred_idx]

  return {
      "prediction_index": int(pred_idx),
      "prediction_name": str(pred_name)
  }

# ──────────────────────────────────────────────────────────────────────────────
# 5. Baseline model endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict/baseline", summary="Predict with baseline model")
def predict_baseline(data: InputData):      ### as requested in step 3.
  return do_predict(baseline_model, data)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Augmented model endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict/augmented", summary="Predict with augmented model")
def predict_augmented(data: InputData):
  return do_predict(augmented_model, data)
