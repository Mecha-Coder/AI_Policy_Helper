#!/bin/sh
set -e

echo "⏳ Backend waiting for dependencies… (120s)"
sleep 120

echo "🚀 Backend starting now!"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000