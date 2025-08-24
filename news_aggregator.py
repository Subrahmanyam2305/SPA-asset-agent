import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from mixrank_data import MixRank
from yfinance_data import YFinance

class ResearchDocs:
    def __init__(self, db_path: str = "research_docs.db"):
        """
        Initialize the ResearchDocs class with a SQLite database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        self.logger = logging.getLogger(__name__)

    def _init_db(self) -> None:
        """Initialize the SQLite database with the research_docs table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            raise

    def _needs_update(self, last_updated: str) -> bool:
        """
        Check if the asset needs to be updated based on the 5-day threshold.
        
        Args:
            last_updated (str): ISO format timestamp string
            
        Returns:
            bool: True if asset needs update, False otherwise
        """
        try:
            last_update_date = datetime.fromisoformat(last_updated)
            five_days_ago = datetime.now() - timedelta(days=5)
            return last_update_date < five_days_ago
        except ValueError as e:
            self.logger.error(f"Error parsing date: {e}")
            return True

    def get_research_updates(self, asset_name: str) -> Dict[str, Any]:
        """
        Get research updates for a specific asset. If the asset hasn't been initialized
        or needs an update (>5 days old), it will make a new call to fetch and store the data.
        
        Args:
            asset_name (str): Name of the asset
            
        Returns:
            Dict containing the research data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if asset exists and get its data
                cursor.execute(
                    "SELECT last_updated, mixrank_content, yfinance_content FROM research_docs WHERE asset_name = ?",
                    (asset_name,)
                )
                result = cursor.fetchone()
                
                if result is None or (result and self._needs_update(result[0])):
                    # Asset not initialized or needs update
                    return self._initialize_asset(asset_name)
                else:
                    # Return existing data
                    return {
                        "asset_name": asset_name,
                        "last_updated": result[0],
                        "mixrank_content": result[1],
                        "yfinance_content": result[2]
                    }
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching research updates for asset {asset_name}: {e}")
            raise

    def _initialize_asset(self, asset_name: str) -> Dict[str, Any]:
        """
        Initialize a new asset in the database with initial research data.
        This method should be extended to include actual API calls for mixrank and yfinance.
        
        Args:
            asset_name (str): Name of the asset
            
        Returns:
            Dict containing the research data
        """
        try:
            current_time = datetime.now().isoformat()
            
            # TODO: Add actual API calls to mixrank and yfinance here
            # For now, we'll just create placeholder entries
            mixrank_data = f"Placeholder Mixrank data for {asset_name}"
            yfinance_data = f"Placeholder YFinance data for {asset_name}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update asset record
                cursor.execute("""
                    INSERT OR REPLACE INTO research_docs 
                    (asset_name, last_updated, mixrank_content, yfinance_content)
                    VALUES (?, ?, ?, ?)
                """, (asset_name, current_time, mixrank_data, yfinance_data))
                
                conn.commit()
                
                return {
                    "asset_name": asset_name,
                    "last_updated": current_time,
                    "mixrank_content": mixrank_data,
                    "yfinance_content": yfinance_data
                }
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing asset {asset_name}: {e}")
            raise

    def update_asset(self, asset_name: str, mixrank_content: str = None, yfinance_content: str = None) -> None:
        """
        Update the research content for an asset.
        
        Args:
            asset_name (str): Name of the asset
            mixrank_content (str, optional): New Mixrank content
            yfinance_content (str, optional): New YFinance content
        """
        try:
            current_time = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing data first
                cursor.execute(
                    "SELECT mixrank_content, yfinance_content FROM research_docs WHERE asset_name = ?",
                    (asset_name,)
                )
                result = cursor.fetchone()
                
                # Use existing content if new content is not provided
                if result:
                    existing_mixrank, existing_yfinance = result
                    mixrank_content = mixrank_content if mixrank_content is not None else existing_mixrank
                    yfinance_content = yfinance_content if yfinance_content is not None else existing_yfinance
                
                # Insert or update the record
                cursor.execute("""
                    INSERT OR REPLACE INTO research_docs 
                    (asset_name, last_updated, mixrank_content, yfinance_content)
                    VALUES (?, ?, ?, ?)
                """, (asset_name, current_time, mixrank_content, yfinance_content))
                
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error updating asset {asset_name}: {e}")
            raise


if __name__ == "__main__":
    research_docs = ResearchDocs()
    # Example usage
    asset_companies = ["openai", "google", "microsoft", "amazon", "facebook", "apple", "tesla", "nvidia", "amd", "intel", "digital realty", "equinix"]
    for company in asset_companies:
        mixrank_content = MixRank().get_research_updates(company_name=company)
        yfinance_content = YFinance().get_research_updates(company)
        if yfinance_content:  # Only update if we have valid yfinance data
            research_docs.update_asset(company, 
                             mixrank_content=mixrank_content,
                             yfinance_content=yfinance_content)
    
    print(research_docs.get_research_updates("openai"))