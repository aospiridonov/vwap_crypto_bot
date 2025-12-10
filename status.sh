#!/bin/bash
# Check bot status

echo "📊 Indicators Crypto Bot Status"
echo "================================"
docker-compose ps
echo ""
echo "Recent logs (last 20 lines):"
echo "----------------------------"
docker-compose logs --tail=20
