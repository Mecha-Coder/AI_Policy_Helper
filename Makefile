.PHONY: test fmt offline-dev online-dev clean

clean:
	docker-compose -f docker-compose.online.yml down --rmi all -v
	docker-compose -f docker-compose.offline.yml down --rmi all -v

test:
	docker compose run --rm backend pytest -q

fmt:
	docker compose run --rm backend black app

offline:
	sed -i '/^OLLAMA_HOST=/d' .env
	docker-compose -f docker-compose.offline.yml up --build

online:
	sed -i '/^OLLAMA_HOST=/d' .env
	echo "OLLAMA_HOST=http://ollama:11434" >> .env
	docker-compose -f docker-compose.online.yml up --build