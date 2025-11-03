.PHONY: help build up down train collect features predict clean logs

help:
	@echo "Monterey Bay Fishing Prediction - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make build              - Build Docker images"
	@echo ""
	@echo "Training:"
	@echo "  make train              - Train model with tidal data (R²=0.72)"
	@echo "  make train-prod         - Train production model"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collect            - Collect all data (NOAA, weather, tidal)"
	@echo "  make collect-noaa       - Collect NOAA historical data"
	@echo "  make collect-buoy       - Collect buoy temperatures"
	@echo "  make collect-weather    - Collect weather data"
	@echo "  make collect-tidal      - Collect tidal data"
	@echo "  make collect-pressure   - Collect barometric pressure"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline           - Run full enhanced pipeline"
	@echo "  make features           - Generate features only"
	@echo ""
	@echo "Analysis:"
	@echo "  make compare            - Compare models"
	@echo "  make visualize          - Visualize buoy temps"
	@echo "  make visualize-model    - Visualize model behavior"
	@echo ""
	@echo "Prediction:"
	@echo "  make forecast           - Generate fishing forecast"
	@echo "  make predict            - Predict fishing conditions"
	@echo ""
	@echo "Maintenance:"
	@echo "  make logs SERVICE=train - Show logs for a service"
	@echo "  make clean              - Remove containers and volumes"
	@echo "  make down               - Stop all services"

build:
	docker-compose build

train:
	docker-compose run --rm train

train-prod:
	docker-compose --profile production run --rm train-production

collect:
	@echo "Collecting all data..."
	docker-compose --profile data-collection run --rm collect-data
	docker-compose --profile data-collection run --rm collect-buoy
	docker-compose --profile data-collection run --rm collect-weather
	docker-compose --profile data-collection run --rm collect-tidal
	docker-compose --profile data-collection run --rm collect-pressure

collect-noaa:
	docker-compose --profile data-collection run --rm collect-data

collect-buoy:
	docker-compose --profile data-collection run --rm collect-buoy

collect-weather:
	docker-compose --profile data-collection run --rm collect-weather

collect-tidal:
	docker-compose --profile data-collection run --rm collect-tidal

collect-pressure:
	docker-compose --profile data-collection run --rm collect-pressure

features:
	docker-compose --profile pipeline run --rm features

pipeline:
	docker-compose --profile pipeline run --rm pipeline

compare:
	docker-compose --profile analysis run --rm compare

visualize:
	docker-compose --profile analysis run --rm visualize

visualize-model:
	docker-compose --profile analysis run --rm visualize-model

forecast:
	docker-compose --profile prediction run --rm forecast

predict:
	docker-compose --profile prediction run --rm predict

logs:
	@if [ -z "$(SERVICE)" ]; then \
		docker-compose logs -f; \
	else \
		docker-compose logs -f $(SERVICE); \
	fi

down:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

up:
	docker-compose up

shell:
	docker-compose run --rm train /bin/bash
