PY ?= .venv/bin/python
.PHONY: bootstrap smoke test
bootstrap:
	./bootstrap.sh
smoke:            ## M0 stub end-to-end + exact scorecard assertion + full pytest (no network, no keys, no spend)
	PY=$(PY) scripts/smoke.sh
test:
	PYTHONPATH=. $(PY) -m pytest -q
