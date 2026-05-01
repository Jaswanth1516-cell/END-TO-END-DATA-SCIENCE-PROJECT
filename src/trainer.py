from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_preprocessing import build_feature_matrix, load_data

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "iris.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"


def train_model() -> None:
    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    X, y = build_feature_matrix(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    print("Validation classification report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    metadata = {
        "pipeline": pipeline,
        "feature_names": X.columns.tolist(),
        "target_names": ["setosa", "versicolor", "virginica"],
    }
    joblib.dump(metadata, MODEL_PATH)
    print(f"Saved trained model and preprocessing pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
