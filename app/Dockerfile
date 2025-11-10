# app/Dockerfile
FROM python:3.13.7

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Copy the application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["python", "training_model.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
