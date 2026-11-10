.PHONY: install test lint api dashboard simulate backtest
install:   ; pip install -e ".[dev]" && pip install -r dashboard/requirements.txt
test:      ; python -m pytest -q
lint:      ; ruff check src tests
simulate:  ; python -m edgeagent.cli simulate --hours 480 --step 3
backtest:  ; python -m edgeagent.cli backtest
api:       ; python -m edgeagent.cli serve
dashboard: ; cd dashboard && EDGEAGENT_API_URL=http://localhost:8000 streamlit run app.py
