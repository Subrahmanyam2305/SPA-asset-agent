import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from mixrank_data import MixRank
from yfinance_data import YFinance
from graph_data import get_model_recommendations, StockGraphModel, load_data, prepare_datasets, train_model
import torch
import yfinance as yf


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
                        yfinance_content TEXT,
                        graph_content TEXT
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
                    "SELECT last_updated, mixrank_content, yfinance_content, graph_content FROM research_docs WHERE asset_name = ?",
                    (asset_name,)
                )
                result = cursor.fetchone()
                
                if result is None or (result and self._needs_update(result[0])):
                    # Asset not initialized or needs update
                    return self._initialize_asset(asset_name)
                else:
                    # Return existing data with deserialized graph content
                    graph_content = json.loads(result[3]) if result[3] else None
                    return {
                        "asset_name": asset_name,
                        "last_updated": result[0],
                        "mixrank_content": result[1],
                        "yfinance_content": result[2],
                        "graph_content": graph_content
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
            
            # Get data from each source
            mixrank_data = MixRank().get_research_updates(company_name=asset_name)
            yfinance_data = YFinance().get_research_updates(asset_name)
            
            # Get graph model predictions
            tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
            window_size = 20
            input_size = 7 * window_size
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get price data
            prices, returns = load_data(start_date="2023-01-01", end="2025-01-01")
            
            # Initialize model
            model = StockGraphModel(
                in_channels=input_size,
                hidden_channels=64,
                out_channels=2
            ).to(device)
            
            # Check if model exists, if not train it
            model_path = Path('best_model.pt')
            if not model_path.exists():
                print("No trained model found. Training new model...")
                # Prepare datasets for training
                datasets = prepare_datasets(returns, window_size)
                # Train the model
                model, best_val_acc = train_model(
                    *datasets[:8],  # First 8 elements are train and val data
                    window_size=window_size,
                    device=device
                )
                print(f"Model trained successfully. Best validation accuracy: {best_val_acc:.4f}")
            else:
                # Load existing model
                model.load_state_dict(torch.load('best_model.pt'))
                model = model.to(device)
                model.eval()
            graph_data = get_model_recommendations(model, prices)
            
            # Convert graph_data to JSON string for storage
            graph_data_str = json.dumps(graph_data) if graph_data is not None else None
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update asset record
                cursor.execute("""
                    INSERT OR REPLACE INTO research_docs 
                    (asset_name, last_updated, mixrank_content, yfinance_content, graph_content)
                    VALUES (?, ?, ?, ?, ?)
                """, (asset_name, current_time, mixrank_data, yfinance_data, graph_data_str))
                
                conn.commit()
                
                return {
                    "asset_name": asset_name,
                    "last_updated": current_time,
                    "mixrank_content": mixrank_data,
                    "yfinance_content": yfinance_data,
                    "graph_content": graph_data
                }
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing asset {asset_name}: {e}")
            raise

    def update_asset(self, asset_name: str, mixrank_content: str = None, yfinance_content: str = None, graph_content: Dict = None) -> None:
        """
        Update the research content for an asset.
        
        Args:
            asset_name (str): Name of the asset
            mixrank_content (str, optional): New Mixrank content
            yfinance_content (str, optional): New YFinance content
            graph_content (Dict, optional): New Graph content (will be JSON serialized)
        """
        try:
            current_time = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing data first
                cursor.execute(
                    "SELECT mixrank_content, yfinance_content, graph_content FROM research_docs WHERE asset_name = ?",
                    (asset_name,)
                )
                result = cursor.fetchone()
                
                # Use existing content if new content is not provided
                if result:
                    existing_mixrank, existing_yfinance, existing_graph = result
                    mixrank_content = mixrank_content if mixrank_content is not None else existing_mixrank
                    yfinance_content = yfinance_content if yfinance_content is not None else existing_yfinance
                    if graph_content is None:
                        graph_content = json.loads(existing_graph) if existing_graph else None
                
                # Convert graph_content to JSON string if it's a dictionary
                graph_content_str = json.dumps(graph_content) if graph_content is not None else None
                
                # Insert or update the record
                cursor.execute("""
                    INSERT OR REPLACE INTO research_docs 
                    (asset_name, last_updated, mixrank_content, yfinance_content, graph_content)
                    VALUES (?, ?, ?, ?, ?)
                """, (asset_name, current_time, mixrank_content, yfinance_content, graph_content_str))
                
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error updating asset {asset_name}: {e}")
            raise


if __name__ == "__main__":
    research_docs = ResearchDocs()
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
    window_size = 20
    input_size = 7* window_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockGraphModel(
        in_channels=input_size,
        hidden_channels=64,  # Reduced from 128
        out_channels=2
    ).to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    model = model.to(device)
    model.eval()
    prices = yf.download(tickers, start="2023-01-01", end="2025-01-01")['Close']
    # Example usage
    asset_companies = ["openai", "google", "microsoft", "amazon", "facebook", "apple", "tesla", "nvidia", "amd", "intel"]
    for company in asset_companies:
        mixrank_content = MixRank().get_research_updates(company_name=company)
        yfinance_content = YFinance().get_research_updates(company)
        # load the graph model from best_model.pt
        graph_content = get_model_recommendations(model, prices)
        if yfinance_content:  # Only update if we have valid yfinance data
            research_docs.update_asset(company, 
                             mixrank_content=mixrank_content,
                             yfinance_content=yfinance_content,
                             graph_content=graph_content)
    
    print(research_docs.get_research_updates("openai"))