FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model_trainer.py .
COPY flask_app.py .
COPY fastapi_app.py .
COPY model.pkl .
COPY scaler.pkl .

# Expose ports
EXPOSE 5000 8000

# Default to Flask
CMD ["python", "flask_app.py"]
