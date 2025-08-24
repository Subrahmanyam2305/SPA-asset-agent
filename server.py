from __future__ import annotations

import os
import threading
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import sqlite3


# ---------------------------
# Data models
# ---------------------------
RiskLevel = str


class Asset(BaseModel):
    name: str
    risks: RiskLevel
    details: str


class AssetInput(BaseModel):
    name: str


class AddAssetRequest(BaseModel):
    asset: AssetInput


class PortfolioResponse(BaseModel):
    portfolio: List[Asset]


# ---------------------------
# App + CORS
# ---------------------------
load_dotenv()

app = FastAPI(title="SPA Asset Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# DB helpers
# ---------------------------
DB_PATH_ENV_KEYS = ("RESEARCH_DB_PATH", "DB_PATH", "DATABASE_URL")


def _get_db_path() -> str:
    for key in DB_PATH_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return "research_docs.db"


def _ensure_db_initialized() -> None:
    try:
        # Lazily import to avoid hard dependency during module import
        import init as _db_init  # type: ignore

        _db_init.create_tables(_get_db_path())
    except Exception as exc:
        print(f"Database initialization check failed: {exc}")


def _fetch_portfolio_from_db() -> List[Asset]:
    db_path = _get_db_path()
    assets: List[Asset] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT d.asset_name, r.report, r.risk_level
                FROM research_docs d
                LEFT JOIN research_reports r ON r.asset_name = d.asset_name
                ORDER BY d.asset_name
                """
            )
            rows = cursor.fetchall()
            for name, report, risk_level in rows:
                details = (
                    report
                    if isinstance(report, str) and report.strip()
                    else "No details yet. Use refresh to update risk assessment and commentary."
                )
                # Use stored DB value directly; default to "none" when missing/empty
                risk_out: RiskLevel = (
                    risk_level if isinstance(risk_level, str) and risk_level.strip() else "none"
                )
                assets.append(Asset(name=name, risks=risk_out, details=details))
    except Exception as exc:
        print(f"Failed to fetch portfolio from DB: {exc}")
    return assets


# ---------------------------
# Routes
# ---------------------------
@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio() -> PortfolioResponse:
    _ensure_db_initialized()
    assets = _fetch_portfolio_from_db()
    return PortfolioResponse(portfolio=assets)


@app.post("/addasset", response_model=Asset, status_code=201)
async def add_asset(payload: AddAssetRequest) -> Asset:
    stored_name = payload.asset.name

    # Persist to DB (case-insensitive uniqueness)
    try:
        _ensure_db_initialized()
        with sqlite3.connect(_get_db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT asset_name FROM research_docs WHERE LOWER(asset_name)=LOWER(?)",
                (stored_name,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO research_docs (asset_name) VALUES (?)",
                    (stored_name,),
                )
                conn.commit()
            else:
                stored_name = row[0]
    except Exception as exc:
        print(f"Failed to persist asset to DB: {exc}")

    new_asset = Asset(
        name=stored_name,
        risks="none",
        details=(
            "No details yet. Use refresh to update risk assessment and commentary."
        ),
    )
    return new_asset


@app.post("/refresh")
async def refresh_portfolio() -> dict:
    # Placeholder: hook your analysis/ETL pipeline here to update risks/details
    # For the stub, we simply acknowledge the request.
    return {"status": "ok"}


# ---------------------------
# Root
# ---------------------------
@app.get("/")
async def root() -> dict:
    return {"message": "SPA Asset Agent API is running"}


# ---------------------------
# ngrok integration (optional)
# ---------------------------
def _start_ngrok(port: int) -> str | None:
    try:
        from pyngrok import ngrok

        token = os.getenv("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)

        domain = os.getenv("NGROK_DOMAIN") or os.getenv("NGROK_HOSTNAME")
        endpoint_id = os.getenv("NGROK_ENDPOINT_ID")
        region = os.getenv("NGROK_REGION")

        connect_kwargs = {"addr": port, "proto": "http"}
        if region:
            connect_kwargs["region"] = region

        tunnel = None
        if domain:
            # Try hostname kw (v2), then domain kw (v3)
            try:
                tunnel = ngrok.connect(**{**connect_kwargs, "hostname": domain})
                print(f"ngrok bound to domain: {domain}")
            except TypeError:
                try:
                    tunnel = ngrok.connect(**{**connect_kwargs, "domain": domain})
                    print(f"ngrok bound to domain: {domain}")
                except Exception as bind_exc:
                    print(f"Failed to bind ngrok to domain {domain}: {bind_exc}")
            except Exception as bind_exc:
                print(f"Failed to bind ngrok to domain {domain}: {bind_exc}")

        if tunnel is None and endpoint_id:
            try:
                tunnel = ngrok.connect(**{**connect_kwargs, "labels": f"edge={endpoint_id}"})
                print(f"ngrok bound to endpoint id: {endpoint_id}")
            except TypeError:
                print("ngrok agent does not support 'labels' param; falling back to default tunnel")
            except Exception as edge_exc:
                print(f"Failed to bind ngrok to endpoint id {endpoint_id}: {edge_exc}")

        if tunnel is None:
            tunnel = ngrok.connect(**connect_kwargs)

        public_url = tunnel.public_url
        print(f"ngrok tunnel established: {public_url} -> http://127.0.0.1:{port}")
        try:
            with open("ngrok_url.txt", "w") as f:
                f.write(public_url)
        except Exception as write_exc:
            print(f"Failed to write ngrok URL to file: {write_exc}")
        return public_url
    except Exception as exc:
        print(f"ngrok not started: {exc}")
        return None


if __name__ == "__main__":
    PORT = 8080
    # _start_ngrok(PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)