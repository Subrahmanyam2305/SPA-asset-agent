import logging
import sqlite3

def create_tables(db_path: str) -> None:
    """Initialize the SQLite database with tables."""

    logger = logging.getLogger(__name__)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create single table for asset research data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_docs (
                    asset_name TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    mixrank_content TEXT,
                    yfinance_content TEXT
                )
            """)

            # Create single table with reports
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_reports (
                    asset_name TEXT PRIMARY KEY,
                    risk_level VARCHAR,
                    report TEXT
                )
            """)
            
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

def populate_asset(db_path: str, assets: list[str]):
    logger = logging.getLogger(__name__)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            for a in assets:
                cursor.execute(
                    "SELECT 1 FROM research_docs WHERE asset_name = ?",
                    (a,)
                )
                if cursor.fetchone() is None:
                    cursor.execute(
                        "INSERT INTO research_docs (asset_name) VALUES (?)",
                        (a,)
                    )
            
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

def init_db(db_name: str):
    create_tables(db_name)
    populate_asset(db_name,
        ["openai", "google", "microsoft", "amazon", "facebook", "apple", "tesla", "nvidia", "amd", "intel", "digital realty", "equinix"])