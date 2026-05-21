"""
Test script for API endpoints
Usage: python test_api.py
"""

import requests
import json
from typing import Dict, Any

# Change this to your API URL
API_URL = "http://localhost:5000"  # For Flask
# API_URL = "http://localhost:8000"  # For FastAPI

def print_response(title: str, response: Dict[Any, Any]):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(response, indent=2))

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print_response("Health Check", response.json())
    assert response.status_code == 200

def test_features():
    """Test features endpoint"""
    response = requests.get(f"{API_URL}/features")
    print_response("Features", response.json())
    assert response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{API_URL}/model-info")
    print_response("Model Info", response.json())
    assert response.status_code == 200

def test_predict_single():
    """Test single prediction"""
    # Create a sample with 10 features
    features = [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01]
    
    payload = {"features": features}
    response = requests.post(f"{API_URL}/predict", json=payload)
    print_response("Single Prediction", response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_batch():
    """Test batch prediction"""
    features = [
        [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01],
        [0.1, -0.1, 0.05, -0.03, 0.02, -0.05, 0.03, -0.04, 0.05, -0.02],
        [-0.05, 0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.03, -0.04, 0.01]
    ]
    
    payload = {"features": features}
    response = requests.post(f"{API_URL}/predict-batch", json=payload)
    print_response("Batch Prediction", response.json())
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3

def test_invalid_input():
    """Test with invalid input"""
    payload = {"features": [1, 2, 3]}  # Wrong number of features
    response = requests.post(f"{API_URL}/predict", json=payload)
    print_response("Invalid Input (Expected Error)", response.json())
    assert response.status_code == 400

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Data Science ML API")
    print("="*60)
    
    try:
        test_health()
        test_features()
        test_model_info()
        test_predict_single()
        test_predict_batch()
        test_invalid_input()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Cannot connect to API at {API_URL}")
        print("Make sure the API server is running:")
        print("  Flask: python flask_app.py")
        print("  FastAPI: uvicorn fastapi_app:app --reload")
    
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
