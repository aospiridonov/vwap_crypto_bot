#!/bin/bash
# Rebuild and restart the bot

echo "🔨 Rebuilding Indicators Crypto Bot..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "✅ Bot rebuilt and started!"
echo "📊 View logs: ./logs.sh"
