PIP := pip3
RUFF := ruff

.PHONY: all
all: req lint clean

.PHONY: clean
clean: ## Clean up generated python files
	rm -rf src/rag/__pycache__

.PHONY: lint
lint: ## Run code formatter and linter (using ruff)
	$(RUFF) check src/rag/. --fix

.PHONY: req
req: ## Install the python requirements
	$(PIP) install -r requirements.txt
