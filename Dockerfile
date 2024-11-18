# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install system dependencies required for ChromaDB and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Install dependencies from Pipfile.lock
RUN pipenv install --deploy --system

# Copy the application's files
COPY ["app.py", "create_collection.py", "delete_collection.py", "upload_images.py", "./"]

# Create and set permissions for the database directory
RUN mkdir -p my_vectordb && chmod 777 my_vectordb

# Expose Gradio port
EXPOSE 7860

# Set Gradio server settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app.py"]