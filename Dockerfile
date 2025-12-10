FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Set Python path
ENV PYTHONPATH=/app

# Run the VWAP bot
CMD ["python", "trading_bot/main.py"]
