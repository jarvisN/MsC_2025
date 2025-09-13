# Use the official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application files (including models)
COPY app.py .
COPY models/ models/

# Expose the port the app will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]