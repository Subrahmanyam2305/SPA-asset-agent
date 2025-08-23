from dataclasses import dataclass

@dataclass
class Asset:
    name: str
    ticker: str | None

@dataclass
class ResearchUpdates: # base class to extend
    relevance_reasoning: str

class ResearchSource:
    def research_asset_update(self, asset: Asset) -> ResearchUpdates:
        raise NotImplemented()
    
    def initialized() -> bool:
        raise NotImplemented()
    
    def initialize_asset(self, asset: Asset) -> None:
        raise NotImplemented()