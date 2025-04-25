FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y libomp-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Run the app
CMD ["python", "app.py"]
