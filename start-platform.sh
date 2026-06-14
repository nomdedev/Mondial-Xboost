#!/bin/bash
# start-platform.sh - Start the complete Mondial-Xboost platform

cd "$(dirname "$0")"

echo "=== Mondial-Xboost Platform ==="
echo "Starting services..."

# Start API server
source venv/Scripts/activate
echo "[1/2] Starting FastAPI server on http://localhost:8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to be ready
sleep 3

# Open dashboard
echo "[2/2] Opening dashboard..."
start dashboard/index.html

echo ""
echo "Platform running:"
echo "  API: http://localhost:8000"
echo "  Dashboard: dashboard/index.html"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

wait $API_PID
