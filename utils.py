"""
Utility functions for the ML API project
"""

import json
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_metadata(metadata_path: str = 'model_metadata.json') -> Dict[str, Any]:
    """Load model metadata"""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Metadata file not found at {metadata_path}")
        return {}

def validate_features(features: List[float], expected_count: int = 10) -> bool:
    """Validate feature list"""
    if not isinstance(features, list):
        return False
    if len(features) != expected_count:
        return False
    return all(isinstance(f, (int, float)) for f in features)

def validate_batch_features(features_batch: List[List[float]], expected_count: int = 10) -> bool:
    """Validate batch of features"""
    if not isinstance(features_batch, list) or len(features_batch) == 0:
        return False
    return all(validate_features(f, expected_count) for f in features_batch)

def format_response(success: bool, data: Dict = None, error: str = None) -> Dict:
    """Format API response"""
    response = {
        'success': success,
        'timestamp': str(np.datetime64('now'))
    }
    
    if data:
        response.update(data)
    if error:
        response['error'] = error
    
    return response

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values"""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }

def log_prediction(features: List[float], prediction: float, model_type: str = "RandomForest"):
    """Log prediction for audit trail"""
    logger.info(f"Prediction made: model={model_type}, prediction={prediction:.4f}")

if __name__ == "__main__":
    # Test utility functions
    test_features = [0.05, -0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.03, 0.04, -0.01]
    print(f"Features valid: {validate_features(test_features)}")
    print(f"Statistics: {calculate_statistics(test_features)}")
