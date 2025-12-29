FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project files
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY README.md .

# Default command to run the training script
CMD ["python", "src/train.py"]
