#!/bin/sh
set -e

if [ -n "$OLLAMA_HOST" ]; then
    echo "⏳ Backend waiting for dependencies… (90s)"
    sleep 90
else
    echo "ℹ️  OLLAMA_HOST not set, skipping delay"
fi

echo "🚀 Backend starting now!"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000