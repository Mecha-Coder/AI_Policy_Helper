#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

MODEL=${MODEL_NAME:-llama3.2:3b}

echo "🔴 Retrieve $MODEL model..."
ollama pull $MODEL
echo "🟢 Done! Model $MODEL is ready."

# Wait for Ollama process to finish.
wait $pid