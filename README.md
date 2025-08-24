> Front-end sources are available here: [Download front-end sources](https://drive.usercontent.google.com/download?id=14c6-9glCokVysyz4J0T-Th6_wVEKfVuz&export=download&authuser=0)

# SPA-asset-agent

AI-powered backend that ingests signals about companies/assets from multiple sources, runs an agentic risk-assessment pipeline, and serves portfolio risk summaries via a REST API.

Built with FastAPI, LangGraph/LangChain, Tavily tools, and SQLite storage. Optional ngrok integration enables secure public tunneling during development.

## Features

- Risk assessment pipeline combining a research agent and a reporting agent
- Connectors to external data sources (SixtyFour, MixRank; optional yfinance helper)
- SQLite-backed storage for assets and generated research reports
- REST API to manage assets and fetch portfolio-level risk summaries
- Optional ngrok support to expose the API for demos

## Architecture

- Backend API: FastAPI app in `server.py`
- Agentic pipeline: LangGraph-based workflow in `agentic_pipeline.py`
  - Risk Assessor: gathers weak signals and optionally calls Tavily search/extract tools
  - Reporter: synthesizes findings into a markdown report with cited sources and a risk level
- Data sources/integrations:
  - `src_sixtyfour.py`: SixtyFour enrich-company API (requires `SIXTYFOUR_API_KEY`)
  - `mixrank_data.py`: MixRank company/employee timeseries (requires `MIXRANK_API_KEY`)
  - `yfinance_data.py`: optional ticker-based financial context (no direct pipeline integration yet)
- Storage: SQLite database with two tables created by `init.py`
  - `research_docs(asset_name, last_updated, mixrank_content, yfinance_content)`
  - `research_reports(asset_name, risk_level, report)`

## Quickstart

### Prerequisites
- Python 3.11+
- A `.env` file with required API keys (see below)

### Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment
Create a `.env` file in the project root:
```
# LLMs & tools
OPENAI_API_KEY=...          # required by langchain init_chat_model("openai:gpt-5")
TAVILY_API_KEY=...          # required for TavilySearch/TavilyExtract tools

# Data sources
SIXTYFOUR_API_KEY=...
MIXRANK_API_KEY=...

# Database (optional override)
RESEARCH_DB_PATH=research_docs.db

# ngrok (optional)
NGROK_AUTHTOKEN=...
NGROK_DOMAIN=...            # or NGROK_HOSTNAME
NGROK_ENDPOINT_ID=...
NGROK_REGION=us
```

### Initialize DB and generate reports
This seeds default assets and runs the agentic pipeline to populate `research_reports`:
```
python application.py
```

### Run the API server
```
python server.py
```
The server listens on `http://127.0.0.1:8080` by default.

To expose it via ngrok, set the ngrok env vars above and optionally enable `_start_ngrok(PORT)` in `server.py`.

## API

### GET /
Health check.

Response:
```
{ "message": "SPA Asset Agent API is running" }
```

### GET /portfolio
Returns current portfolio with risk summaries pulled from the database.

Response shape:
```
{
  "portfolio": [
    {
      "name": "google",
      "risks": "medium",     
      "details": "...markdown report..."
    }
  ]
}
```

### POST /addasset
Adds an asset (case-insensitive uniqueness) to `research_docs`. Risk/details remain empty until a refresh run populates them.

Request:
```
{
  "asset": { "name": "acme corp" }
}
```

Response:
```
{
  "name": "acme corp",
  "risks": "none",
  "details": "No details yet. Use refresh to update risk assessment and commentary."
}
```

### POST /refresh
Placeholder endpoint for triggering refresh logic. Returns `{ "status": "ok" }`.

## How it works

1. `init.py` creates the SQLite schema and seeds a default list of well-known assets.
2. `application.py` loads assets and sources, runs the agentic pipeline per asset, and writes results into `research_reports` (including `risk_level`).
3. `server.py` reads from the database to serve `/portfolio`, and handles `/addasset` and `/refresh`.

## Repository layout

- `server.py`: FastAPI app and routes
- `application.py`: batch update job to run the pipeline and persist reports
- `agentic_pipeline.py`: LangGraph pipeline (Risk Assessor + Reporter)
- `src_base.py`: shared data classes and base interfaces
- `src_sixtyfour.py`: SixtyFour integration
- `mixrank_data.py`: MixRank integration
- `yfinance_data.py`: optional yfinance helper
- `init.py`: database creation and seeding
- `data/`: cached or auxiliary data

## License
See `LICENSE` for details.

## Acknowledgements
- LangChain & LangGraph
- Tavily
- FastAPI