from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

data_dir = Path(__file__).resolve().parents[1] / "data"
data_dir.mkdir(parents=True, exist_ok=True)


def prepare_iris_data() -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [*iris.feature_names, "target"]
    df.to_csv(data_dir / "iris.csv", index=False)
    print(f"Saved iris dataset to {data_dir / 'iris.csv'}")


if __name__ == "__main__":
    prepare_iris_data()
