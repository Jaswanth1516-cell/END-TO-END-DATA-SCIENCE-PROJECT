from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "iris_model.joblib"


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found. Run `python src/trainer.py` first to train the model: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def predict_species(features: dict) -> dict:
    model_data = load_model()
    pipeline = model_data["pipeline"]
    target_names = model_data["target_names"]
    feature_names = model_data["feature_names"]

    input_df = pd.DataFrame([features], columns=feature_names)
    predicted = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0].tolist()

    return {
        "predicted_class": int(predicted),
        "predicted_label": target_names[int(predicted)],
        "probabilities": {
            target_names[i]: float(probabilities[i]) for i in range(len(probabilities))
        },
    }
