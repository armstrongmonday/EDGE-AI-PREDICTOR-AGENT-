# EdgeAgent Dashboard (front end)
Streamlit UI from README v2: upcoming predictions, match detail, Brainbox health, data quality, backtest, Aviator Lab.
```
pip install -r requirements.txt
EDGEAGENT_API_URL=http://localhost:8000 streamlit run app.py
```
Without a backend it shows synthetic demo data (clearly flagged). Expected API: /v1/fixtures/upcoming, /v1/brainbox/status, /v1/data/sources, /v1/backtest/latest.
