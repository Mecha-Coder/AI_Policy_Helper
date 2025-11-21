.PHONY: dev down test fmt

dev:
	docker compose up --build

clean:
	docker-compose down --rmi all -v

test:
	docker compose run --rm backend pytest -q

fmt:
	docker compose run --rm backend black app
