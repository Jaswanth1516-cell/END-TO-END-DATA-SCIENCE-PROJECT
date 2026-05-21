"""
Comprehensive example showing how to use the ML API
Run this after starting the API server
"""

import requests
import json
import numpy as np
from typing import List, Dict, Any
import time

class MLAPIClient:
    """Client for interacting with ML API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_features(self) -> List[str]:
        """Get feature information"""
        response = self.session.get(f"{self.base_url}/features")
        data = response.json()
        return data.get('features', [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/model-info")
        return response.json()
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make single prediction"""
        payload = {"features": features}
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        return response.json()
    
    def predict_batch(self, features_list: List[List[float]]) -> Dict[str, Any]:
        """Make batch predictions"""
        payload = {"features": features_list}
        response = self.session.post(f"{self.base_url}/predict-batch", json=payload)
        return response.json()

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def example_basic_usage():
    """Example 1: Basic usage"""
    print_section("EXAMPLE 1: Basic API Usage")
    
    client = MLAPIClient()
    
    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model Loaded: {health['model_loaded']}")
    
    # Get features
    print("\n2. Feature Information:")
    features = client.get_features()
    print(f"   Number of features: {len(features)}")
    print(f"   Features: {', '.join(features[:3])}...")
    
    # Get model info
    print("\n3. Model Information:")
    model_info = client.get_model_info()
    print(f"   Model Type: {model_info['model_type']}")
    print(f"   Dataset: {model_info['dataset']}")

def example_single_prediction():
    """Example 2: Single prediction"""
    print_section("EXAMPLE 2: Single Prediction")
    
    client = MLAPIClient()
    
    # Create sample features
    features = [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01]
    
    print(f"Input features: {features}")
    
    # Make prediction
    result = client.predict(features)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nPrediction: {result['prediction']:.2f}")
        print(f"Model Type: {result['model_type']}")

def example_batch_prediction():
    """Example 3: Batch predictions"""
    print_section("EXAMPLE 3: Batch Predictions")
    
    client = MLAPIClient()
    
    # Create multiple samples
    samples = [
        [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01],
        [0.1, -0.1, 0.05, -0.03, 0.02, -0.05, 0.03, -0.04, 0.05, -0.02],
        [-0.05, 0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.03, -0.04, 0.01],
    ]
    
    print(f"Number of samples: {len(samples)}")
    
    # Make batch predictions
    result = client.predict_batch(samples)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nPredictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"   Sample {i}: {pred:.2f}")

def example_random_predictions():
    """Example 4: Random predictions"""
    print_section("EXAMPLE 4: Random Data Predictions")
    
    client = MLAPIClient()
    
    print("Generating random predictions...\n")
    
    predictions = []
    for i in range(5):
        # Generate random features (standardized)
        random_features = list(np.random.randn(10) * 0.5)
        result = client.predict(random_features)
        
        if 'error' not in result:
            predictions.append(result['prediction'])
            print(f"Sample {i+1}: {result['prediction']:.2f}")
    
    if predictions:
        print(f"\nStatistics:")
        print(f"  Mean: {np.mean(predictions):.2f}")
        print(f"  Std:  {np.std(predictions):.2f}")
        print(f"  Min:  {np.min(predictions):.2f}")
        print(f"  Max:  {np.max(predictions):.2f}")

def example_performance_test():
    """Example 5: Performance testing"""
    print_section("EXAMPLE 5: Performance Testing")
    
    client = MLAPIClient()
    
    # Single prediction performance
    features = [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01]
    
    print("Single Prediction Performance:")
    times = []
    for _ in range(10):
        start = time.time()
        client.predict(features)
        times.append(time.time() - start)
    
    print(f"  Average time: {np.mean(times)*1000:.2f}ms")
    print(f"  Min time:     {np.min(times)*1000:.2f}ms")
    print(f"  Max time:     {np.max(times)*1000:.2f}ms")
    
    # Batch prediction performance
    print("\nBatch Prediction Performance (100 samples):")
    batch = [list(np.random.randn(10) * 0.5) for _ in range(100)]
    
    start = time.time()
    result = client.predict_batch(batch)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Per sample: {elapsed*1000/100:.2f}ms")

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("  ML API USAGE EXAMPLES")
    print("="*70)
    print("\nMake sure the API is running:")
    print("  Flask: python flask_app.py")
    print("  FastAPI: uvicorn fastapi_app:app --reload")
    
    try:
        # Run examples
        example_basic_usage()
        example_single_prediction()
        example_batch_prediction()
        example_random_predictions()
        example_performance_test()
        
        print_section("ALL EXAMPLES COMPLETED SUCCESSFULLY ✅")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API server is running!")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()
