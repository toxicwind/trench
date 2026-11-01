.PHONY: render test all
render: ; PYTHONPATH=src python -m trench.render
test: ; ruff check src tests && pytest -q
all: render test
