"""
Configuration module for the ML API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask Configuration
class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

# Model Configuration
class ModelConfig:
    """Model configuration"""
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')
    METADATA_PATH = os.getenv('METADATA_PATH', 'model_metadata.json')
    NUM_FEATURES = 10
    DATASET_TYPE = 'diabetes'

# API Configuration
class APIConfig:
    """API configuration"""
    HOST = os.getenv('API_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', 8000))
    MAX_BATCH_SIZE = 1000
    TIMEOUT = 30

# Logging Configuration
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Get current environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    config = ProductionConfig
elif ENVIRONMENT == 'testing':
    config = TestingConfig
else:
    config = DevelopmentConfig

# Export all configs
__all__ = ['config', 'ModelConfig', 'APIConfig', 'LoggingConfig']
