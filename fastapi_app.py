"""
FastAPI for Model Deployment
Run: uvicorn fastapi_app:app --reload
Access: http://localhost:8000
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import joblib
import os
from model_trainer import ModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Data Science ML API",
    description="End-to-End Data Science Project API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
trainer = None
feature_names = None

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of 10 features")

class BatchPredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., description="List of feature lists")

class PredictionResponse(BaseModel):
    prediction: float
    features_used: int
    model_type: str

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    num_samples: int
    features_per_sample: int

class ModelInfoResponse(BaseModel):
    model_type: str
    num_features: int
    feature_names: List[str]
    dataset: str
    hyperparameters: Dict

def load_model_and_scaler():
    """Load pre-trained model and scaler"""
    global model, scaler, trainer, feature_names
    
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        print("Loading pre-trained model and scaler...")
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        trainer = ModelTrainer()
        _, _ = trainer.load_data('diabetes')
        feature_names = trainer.feature_names
    else:
        print("Model not found. Training new model...")
        trainer = ModelTrainer()
        trainer.train_pipeline(dataset_type='diabetes', save=True)
        model = trainer.model
        scaler = trainer.scaler
        feature_names = trainer.feature_names

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model_and_scaler()
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Home endpoint"""
    return {
        "message": "Welcome to Data Science ML API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/features")
async def get_features():
    """Get feature information"""
    return {
        "features": list(feature_names),
        "num_features": len(feature_names),
        "description": "Diabetes dataset features"
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get detailed model information"""
    return {
        "model_type": "RandomForestRegressor",
        "num_features": 10,
        "feature_names": list(feature_names),
        "dataset": "Diabetes",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 15
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    
    Args:
        features: List of 10 numerical features
    
    Returns:
        Prediction value and metadata
    """
    try:
        if len(request.features) != 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 features, got {len(request.features)}"
            )
        
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        
        return {
            "prediction": float(prediction),
            "features_used": len(request.features),
            "model_type": "RandomForestRegressor"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions
    
    Args:
        features: List of feature lists (each with 10 features)
    
    Returns:
        List of predictions
    """
    try:
        if len(request.features) == 0:
            raise HTTPException(status_code=400, detail="Features list cannot be empty")
        
        for i, sample in enumerate(request.features):
            if len(sample) != 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample {i}: Expected 10 features, got {len(sample)}"
                )
        
        X = np.array(request.features)
        predictions = model.predict(X)
        
        return {
            "predictions": [float(p) for p in predictions],
            "num_samples": len(predictions),
            "features_per_sample": 10
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting FastAPI Server")
    print("="*60)
    print("API available at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
