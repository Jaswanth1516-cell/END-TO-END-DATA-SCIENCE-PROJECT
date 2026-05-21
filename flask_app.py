"""
Flask API for Model Deployment
Run: python flask_app.py
Access: http://localhost:5000
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import json
from model_trainer import ModelTrainer

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
trainer = None
feature_names = None

def load_model_and_scaler():
    """Load pre-trained model and scaler"""
    global model, scaler, trainer, feature_names
    
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        print("Loading pre-trained model and scaler...")
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load trainer to get feature names
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

@app.before_request
def startup():
    """Initialize model on first request"""
    global model, scaler
    if model is None:
        load_model_and_scaler()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Welcome to Data Science ML API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'This help message',
            'GET /health': 'Health check',
            'POST /predict': 'Make predictions',
            'GET /features': 'Get feature information'
        },
        'predict_example': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'features': [0.5, -0.5, 1.2, -0.8, 0.3, -1.0, 0.2, -0.4, 1.5, -0.6]
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get feature information"""
    return jsonify({
        'features': list(feature_names),
        'num_features': len(feature_names),
        'description': 'Diabetes dataset features'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions
    Expected JSON: {'features': [list of 10 numbers]}
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400
        
        features = data['features']
        
        # Validate input
        if not isinstance(features, list):
            return jsonify({'error': '"features" must be a list'}), 400
        
        if len(features) != 10:
            return jsonify({'error': f'Expected 10 features, got {len(features)}'}), 400
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features_used': len(features),
            'model_type': 'RandomForestRegressor'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions
    Expected JSON: {'features': [[list of 10], [list of 10], ...]}
    """
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400
        
        features = data['features']
        
        if not isinstance(features, list) or len(features) == 0:
            return jsonify({'error': 'features must be a non-empty list'}), 400
        
        # Validate each sample
        for i, sample in enumerate(features):
            if len(sample) != 10:
                return jsonify({'error': f'Sample {i}: Expected 10 features, got {len(sample)}'}), 400
        
        # Convert to numpy array
        X = np.array(features)
        
        # Make predictions
        predictions = model.predict(X)
        
        return jsonify({
            'predictions': [float(p) for p in predictions],
            'num_samples': len(predictions),
            'features_per_sample': 10
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'RandomForestRegressor',
        'num_features': 10,
        'feature_names': list(feature_names),
        'dataset': 'Diabetes',
        'train_test_split': '0.8 / 0.2',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 15
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found. See / for available endpoints'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask API Server")
    print("="*60)
    print("Server will be available at: http://localhost:5000")
    print("Visit http://localhost:5000 for documentation")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

    rom flask import Flask, render_template

app = Flask(_name_)

@app.route("/")
def home():
    return render_template("index.html")

if _name_ == "_main_":
    app.run(debug=True)
