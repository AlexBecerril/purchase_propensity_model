from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
CLASSES = list(model.classes_)

app = FastAPI(
    title="Conversion Propensity API",
    description="Customer conversion propensity predictions",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, object]


@app.get("/")
def root():
    return {
        "message": "Conversion Propensity API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/model-info")
def model_info():
    # Support a fitted sklearn Pipeline
    features = getattr(model, "feature_names_in_", None)

    if features is None and hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None:
            features = getattr(
                preprocessor, "feature_names_in_", None
            )

    if features is None:
        features = FEATURES

    # Convert NumPy scalar values into native Python types
    classes = [
        value.item() if hasattr(value, "item") else value
        for value in model.classes_
    ]

    return {
        "model_type": type(model).__name__,
        "pipeline_steps": list(model.named_steps.keys())
            if hasattr(model, "named_steps") else [],
        "features": list(features),
        "classes": classes,
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    missing = [f for f in FEATURES if f not in request.features]
    extra = [f for f in request.features if f not in FEATURES]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        X = pd.DataFrame(
            [[request.features[f] for f in FEATURES]],
            columns=FEATURES,
        )

        probabilities = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]

        probability_by_class = {
            str(cls): float(prob)
            for cls, prob in zip(CLASSES, probabilities)
        }

        conversion_probability = probability_by_class.get("1")

        return {
            "predicted_class": int(prediction),
            "class_probabilities": probability_by_class,
            "conversion_probability": conversion_probability,
            "conversion_probability_percent": (
                round(conversion_probability * 100, 2)
                if conversion_probability is not None
                else None
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc