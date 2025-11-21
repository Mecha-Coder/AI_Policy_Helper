FROM ollama/ollama:latest

COPY ./entrypoint.sh /entrypoint.sh

USER root
RUN chmod +x /entrypoint.sh
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]