.PHONY: test fmt offline-dev online-dev clean

dev:
	docker compose up --build

clean:
	docker-compose down --rmi all -v

test:
	docker compose run --rm backend pytest -q

fmt:
	docker compose run --rm backend black app

offline:
	sed -i '/^OLLAMA_HOST=/d' .env

online:
	sed -i '/^OLLAMA_HOST=/d' .env
	echo "OLLAMA_HOST=http://ollama:11434" >> .env

online-dev: online dev
offline-dev: offline dev