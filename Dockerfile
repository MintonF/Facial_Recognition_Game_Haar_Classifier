FROM tiangolo/opencv-python:3.4.17-python3.8

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install any dependencies (if you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application listens on (if applicable)
EXPOSE 8000

# Command to run the application
CMD ["python", "AppV27AccuracyAndRemovalFalsePositivesV10Rev6BACKUP.py"]

