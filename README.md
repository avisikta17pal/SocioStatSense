# SocioStatSense — Adaptive Real-Time Socio-Economic Modeling

SocioStatSense is a production-ready adaptive statistical modeling system that continuously ingests real-time multi-source socio-economic data streams, updates interpretable models, detects structural shifts, performs causal inference, and provides actionable insights with uncertainty quantification through a modern Streamlit dashboard.

## Features

- Upload & Analyze: drag‑and‑drop CSV upload with strict schema validation
- Modern analytics dashboard: clean, centered UI with interactive Plotly charts
- Fast modeling: multivariate regression for GDP growth with 95% uncertainty bands
- Causal insights: Granger-causality summary and network visualization
- What‑if scenarios: sliders to explore impact of key drivers in real time
- Robust UX: caching, helpful tooltips, and graceful missing‑data handling (imputation)
- DevOps ready: Dockerized services, healthchecks, logs, tests; optional external API clients

## Project Structure

```
.
├── app.py                       # Streamlit UI (modern, centered design)
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile                   # App container image
├── main.py                      # CLI entry points for pipeline/services
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── .env.example                 # Example environment file (copy to .env)
├── .gitignore                   # Ignore secrets, caches, artifacts
├── configs/
│   └── ...                      # Configuration files (YAMLs, etc.)
├── docs/
│   ├── USER_GUIDE.md            # User documentation
│   └── TECHNICAL_DOCUMENTATION.md
├── src/
│   ├── api_clients.py           # Twitter/Alpha Vantage/OpenWeather clients
│   ├── data/                    # Ingestion & preprocessing
│   ├── models/                  # Modeling components
│   ├── causal/                  # Causal inference modules
│   ├── dashboard/               # (If used by main.py dashboard)
│   └── utils/                   # Config/logging/utilities
└── tests/
    ├── unit/                    # Unit tests
    └── integration/             # Integration/E2E tests
```

## Quick Start

Download the dataset:
- View in repo: [socio_economic_synthetic_data.csv](socio_economic_synthetic_data.csv)
  - Open the link and click "Download Raw File" to download the file.


Upload your dataset via the app (File Uploader). The CSV must contain these columns: `date, unemployment_rate, inflation_rate, gdp_growth, twitter_sentiment, google_trends_index, avg_temperature, stock_index_close, mobility_change, covid_cases`. If you want to keep a sample file in the repo, place it at the repository root as `socio_economic_synthetic_data.csv` and upload it from there.

1) Clone and setup

```bash
git clone <your_repo_url>
cd SocioStatSense
```

2) Create .env (API keys)

```bash
cp .env.example .env
# Fill in values for TWITTER_BEARER_TOKEN, ALPHAVANTAGE_API_KEY, OPENWEATHER_API_KEY, etc.
```

3) Install dependencies (local dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4) Run the dashboard (Streamlit)

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

5) Or run with Docker

```bash
docker-compose up --build
# Dashboard service exposes http://localhost:8501
```

## Environment Variables

Place in `.env` (do not commit):

```
TWITTER_API_KEY=
TWITTER_API_SECRET=
TWITTER_ACCESS_TOKEN=
TWITTER_ACCESS_TOKEN_SECRET=
TWITTER_BEARER_TOKEN=
ALPHAVANTAGE_API_KEY=
OPENWEATHER_API_KEY=
```

Docker Compose mounts `.env` into services; locally, `python-dotenv` loads it.

## External API Clients (src/api_clients.py)

- Twitter recent tweets

```python
from src.api_clients import fetch_twitter_recent_tweets

df = fetch_twitter_recent_tweets("inflation", max_results=50)
print(df.head())
```

- Alpha Vantage stock time series

```python
from src.api_clients import fetch_alpha_vantage_stock

df = fetch_alpha_vantage_stock("AAPL", interval="daily")
print(df.tail())
```

- OpenWeather current conditions

```python
from src.api_clients import fetch_openweather_current

data = fetch_openweather_current("London,UK")
print(data["weather"][0]["description"], data["main"]["temp"])
```

The clients use requests/Tweepy with robust error handling, rate-limit detection, and Python logging.

## Dashboard (app.py)

- Modern, centered dark UI with custom CSS
- File uploader with schema validation and clear error messaging
- Time series overview for each column (interactive Plotly)
- Multivariate regression predictions with 95% uncertainty bands
- Granger-causality summary and network visualization
- What‑if scenario sliders with live prediction updates
- Footer with project info/version

## Testing

```bash
pytest tests/ -v
```

## Development

- Use a virtual environment
- Keep secrets in `.env` (see `.gitignore`)
- Format/lint (optional): black, flake8, mypy are included in requirements

## Deployment

- Docker Compose is provided. Update service commands in `docker-compose.yml` as needed.
- Ensure `.env` is provided at runtime (Compose mounts it).

## Troubleshooting

- Missing deps locally? Create `.venv` and `pip install -r requirements.txt`.
- API rate limits? Functions raise clear errors; consider caching/backoff.
- Dashboard not loading? Check terminal logs and port 8501.

## License

MIT License — see LICENSE.
