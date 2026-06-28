# Use lightweight official Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements list
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Generate medical sample scans and model weights placeholder during build
RUN python generate_samples.py && python train.py

# Expose server port
EXPOSE 5000

# Run the Flask app with Gunicorn WSGI server, dynamically binding to $PORT
CMD gunicorn --bind 0.0.0.0:$PORT wsgi:app
