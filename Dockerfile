1. **y the `Dockerfile` to use the `APP_PORT` environment variable:**
``` dockerfile
    FROM tiangolo/opencv-python:3.4.17-python3.8

    WORKDIR /app

    # Install OpenGL
    RUN apt-get update && apt-get install -y libgl1-mesa-glx

    COPY requirements.txt .
    RUN pip install -r requirements.txt

    COPY . .

    # Use APP_PORT environment variable, defaulting to 8080 if not set
    ENV APP_PORT 8080
    CMD ["gunicorn", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6:app", "--bind", "0.0.0.0:$APP_PORT", "--workers", "3"]
```
