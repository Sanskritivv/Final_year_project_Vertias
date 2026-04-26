# 1. Use a slim version of Python to keep the image small and secure
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only the requirements first (optimizes build speed)
COPY requirements.txt .

# 5. Install dependencies
# --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the code
COPY . .

# 7. Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=True
ENV PYTHONPATH=/app/backend

# 8. Start your application 
# Using 1 worker and 2 threads to stay within memory limits (512MB)
CMD gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 app:app
