from fastapi import FastAPI, HTTPException
from app.schemas import IrisFeatures, PredictionResponse
from src.predictor import predict_species

app = FastAPI(
    title="Iris Classification API",
    description="End-to-end data science project: iris data collection, preprocessing, model training, and FastAPI deployment.",
    version="1.0.0",
)


@app.get("/", summary="API status")
def root() -> dict:
    return {
        "service": "Iris Classification API",
        "status": "ready",
        "predict_endpoint": "/predict",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict iris species")
def predict(input_data: IrisFeatures) -> dict:
    features = {
        "sepal length (cm)": input_data.sepal_length_cm,
        "sepal width (cm)": input_data.sepal_width_cm,
        "petal length (cm)": input_data.petal_length_cm,
        "petal width (cm)": input_data.petal_width_cm,
    }

    try:
        result = predict_species(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
