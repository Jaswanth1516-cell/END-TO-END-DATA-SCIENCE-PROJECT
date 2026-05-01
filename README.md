# END-TO-END DATA SCIENCE PROJECT

This repository implements a complete end-to-end data science workflow using the Iris dataset.
It covers data collection, preprocessing, model training, and deployment via a FastAPI service.

## Project Structure

- `scripts/prepare_data.py`: loads iris data and saves a local CSV dataset.
- `src/data_preprocessing.py`: reads the CSV and builds the feature matrix.
- `src/trainer.py`: trains a logistic regression model and saves the pipeline.
- `src/predictor.py`: loads the saved model and runs inference.
- `app/main.py`: FastAPI application exposing a prediction endpoint.

## Setup Instructions

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare the dataset:

```bash
python scripts/prepare_data.py
```

3. Train the model:

```bash
python src/trainer.py
```

4. Start the FastAPI app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Open the interactive API docs:

- http://127.0.0.1:8000/docs

## Example Prediction

Use the `/predict` endpoint with the JSON payload below:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }'
```

Expected response:

```json
{
  "predicted_class": 0,
  "predicted_label": "setosa",
  "probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.00
  }
}
```

## Notes

- The project is intentionally lightweight for easy local deployment.
- The model is stored in `models/iris_model.joblib` after training.
- The API is built with FastAPI for modern REST and auto-generated docs.
