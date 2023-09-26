.PHONY: install all lens fetch

all: fetch lens
	@echo "Fetching your data set and running lens generator"

install:
	poetry run python3 ./scripts/setup.py

fetch: 
	poetry run python3 ./scripts/data_fetcher.py

lens: 
	poetry run python3 ./scripts/lens_generator.py





