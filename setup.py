"""
Setup script for installation
Run: python setup.py install
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-api-project",
    version="1.0.0",
    author="Data Science Team",
    description="End-to-End Data Science Project with API Deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "flask>=2.0.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "joblib>=1.1.0",
        "python-dotenv>=0.19.0",
        "gunicorn>=20.1.0",
    ],
    entry_points={
        "console_scripts": [
            "ml-train=model_trainer:main",
            "ml-flask=flask_app:main",
            "ml-fastapi=fastapi_app:main",
        ],
    },
)
