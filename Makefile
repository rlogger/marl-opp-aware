.PHONY: test compile check check-results deck clean

PYTHON ?= python
NODE ?= node

test:
	$(PYTHON) -m pytest tests

compile:
	$(PYTHON) -m compileall mopa tests

check: test compile
	git diff --check
	$(PYTHON) -m mopa.results --allow-missing

check-results:
	$(PYTHON) -m mopa.results

deck:
	$(NODE) scripts/build_deck.mjs

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
