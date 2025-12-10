#!/bin/bash
# Stop and remove all containers, networks, and volumes

echo "🧹 Cleaning up Indicators Crypto Bot..."
docker-compose down -v

echo "✅ Cleanup complete!"
