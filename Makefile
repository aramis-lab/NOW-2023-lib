POETRY ?= poetry
CONDA ?= conda
CONDA_ENV ?= "./env"

.PHONY: check.lock
check.lock:
	@$(POETRY) check --lock

.PHONY: clean
clean: clean.py clean.test

.PHONY: clean.py
clean.py:
	@find . -name __pycache__ -exec $(RM) -r {} +

.PHONY: clean.test
clean.test:
	@$(RM) -r .pytest cache/

.PHONY: env
env: env.dev

.PHONY: env.conda
env.conda:
	@$(CONDA) env create -p $(CONDA_ENV)

.PHONY: install
install: check.lock
	@$(POETRY) install

.PHONY: test
test: install
	@$(POETRY) run python -m pytest -v tests
