# Use an official Python runtime as a parent image
FROM python:3.9

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set the working directory inside the container
WORKDIR /real-time-face-recognition

# Copy everything from your project folder to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "src/detection.py"]
