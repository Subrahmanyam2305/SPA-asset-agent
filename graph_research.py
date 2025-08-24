from dataclasses import dataclass
from src_base import ResearchSource, ResearchUpdates, Asset
import torch
from graph_data import get_model_recommendations, StockGraphModel, load_data
from pathlib import Path
import json

@dataclass
class GraphResearchUpdates(ResearchUpdates):
    predictions: dict
    model_confidence: dict
    market_context: dict

class GraphResearchSource(ResearchSource):
    def __init__(self):
        self.window_size = 20
        self.input_size = 7 * self.window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or load the model"""
        self.model = StockGraphModel(
            in_channels=self.input_size,
            hidden_channels=64,
            out_channels=2
        ).to(self.device)
        
        model_path = Path('best_model.pt')
        if model_path.exists():
            self.model.load_state_dict(torch.load('best_model.pt'))
            self.model = self.model.to(self.device)
            self.model.eval()

    def name(self) -> str:
        return "GraphResearch"

    def research_asset_update(self, asset: Asset) -> GraphResearchUpdates | None:
        if not self.model:
            return None

        # Load price data
        prices, _ = load_data(start_date="2023-01-01", end="2025-01-01")
        
        # Get model recommendations
        recommendations = get_model_recommendations(self.model, prices)
        
        # Create research updates
        relevance = (
            f"Graph-based market analysis for {asset.name} using deep learning model. "
            f"Model confidence: {recommendations['model_confidence']['average_confidence']:.2f}. "
            f"Market volatility and price changes analyzed for risk assessment."
        )

        return GraphResearchUpdates(
            relevance_reasoning=relevance,
            predictions=recommendations["predictions"],
            model_confidence=recommendations["model_confidence"],
            market_context=recommendations["market_context"]
        )

    def initialized(self) -> bool:
        return self.model is not None

    def initialize_asset(self, asset: Asset) -> None:
        # No per-asset initialization needed
        pass
