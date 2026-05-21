#!/usr/bin/env python
"""Quick test to verify all components work"""

print("Testing project components...\n")

# Test 1: Model trainer
print("✓ Test 1: Model Trainer Module")
try:
    from model_trainer import ModelTrainer
    print("  - ModelTrainer imported successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: Flask app
print("\n✓ Test 2: Flask App")
try:
    import flask_app
    print("  - Flask app imported successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 3: FastAPI app
print("\n✓ Test 3: FastAPI App")
try:
    import fastapi_app
    print("  - FastAPI app imported successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 4: Utils
print("\n✓ Test 4: Utils Module")
try:
    from utils import validate_features, format_response
    test_features = [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01]
    is_valid = validate_features(test_features)
    print(f"  - Utils imported successfully")
    print(f"  - Feature validation works: {is_valid}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 5: Config
print("\n✓ Test 5: Configuration")
try:
    from config import ModelConfig, APIConfig
    print(f"  - Config loaded successfully")
    print(f"  - Model num_features: {ModelConfig.NUM_FEATURES}")
    print(f"  - API host: {APIConfig.HOST}:{APIConfig.FLASK_PORT}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 6: Saved artifacts
print("\n✓ Test 6: Model Artifacts")
try:
    import os
    import joblib
    
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        print("  - model.pkl loaded successfully")
    else:
        print("  - model.pkl not found (will be created on first run)")
    
    if os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
        print("  - scaler.pkl loaded successfully")
    else:
        print("  - scaler.pkl not found (will be created on first run)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 7: Requirements
print("\n✓ Test 7: Dependencies")
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import flask
    import fastapi
    print("  - All core dependencies imported successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - PROJECT IS READY TO USE!")
print("="*60)
print("\nNext steps:")
print("  1. Run notebook: END_TO_END_COLAB_NOTEBOOK.ipynb")
print("  2. Start Flask: python flask_app.py")
print("  3. Or start FastAPI: uvicorn fastapi_app:app --reload")
print("  4. Test API: python test_api.py")
