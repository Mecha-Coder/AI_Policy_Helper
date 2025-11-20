#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &

# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

MODEL_1=${OLLAMA_LLM:-llama3.2:3b}
MODEL_2=${OLLAMA_EMBED:-nomic-embed-text}

echo "🔴 Retrieve model...Please wait for 5mins"
ollama pull $MODEL_1
ollama pull $MODEL_2
echo "🟢 Ollama setup done"

# Wait for Ollama process to finish.
wait $pid