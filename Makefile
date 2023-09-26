.PHONY: install all lens 

all: 
	@echo "Nothing to make"

install:
	poetry run python3 ./scripts/setup.py

fetch: 
	poetry run python3 ./scripts/data_fetcher.py

lens: 
	poetry run python3 ./scripts/lens_generator.py





