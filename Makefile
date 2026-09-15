.PHONY: setup data validate train pipeline test lint clean

PY := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) src/prepare_data.py

validate:
	$(PY) src/validate_data.py

train:
	$(PY) src/train.py

pipeline: data validate train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests conftest.py

clean:
	rm -f data/processed/data.csv model/artifacts/model.pkl
