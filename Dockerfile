FROM python:3.11-slim-bookworm

    # Set the working directory
    WORKDIR /app

    # Copy the requirements file
    COPY requirements.txt .

    # Install dependencies
    RUN pip install --no-cache-dir -r requirements.txt

    # Update package lists and install libgl1-mesa-glx and libglib2.0-0
    RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

    # Copy the application code
    COPY . .

    # Expose port 8000
    EXPOSE 8000

    # Replace 'app' with the actual name of your Flask app instance if it's different
    CMD ["gunicorn", "--bind", "0.0.0.0:8000", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app"]



