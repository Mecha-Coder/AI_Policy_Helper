#!/bin/bash
set -e

MODEL_1=${OLLAMA_LLM:-llama3.2:3b}
MODEL_2=${OLLAMA_EMBED:-nomic-embed-text}

# Start Ollama server in background
/bin/ollama serve &
pid=$!

# Wait a bit for server to initialize
sleep 5

echo "🔴 Pulling models...THIS WILL TAKE SOME TIME"
ollama pull "$MODEL_1"
ollama pull "$MODEL_2"
echo "🟢 Ollama setup complete"

# Keep container running
wait $pid