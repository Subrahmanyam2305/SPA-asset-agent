from dataclasses import dataclass

@dataclass
class Asset:
    name: str
    ticker: str | None

@dataclass
class ResearchUpdates: # base class to extend
    relevance_reasoning: str

class ResearchSource:
    def name(self) -> str:
        return self.__class__.__name__

    def research_asset_update(self, asset: Asset) -> ResearchUpdates | None:
        raise NotImplemented()
    
    def initialized() -> bool:
        raise NotImplemented()
    
    def initialize_asset(self, asset: Asset) -> None:
        raise NotImplemented()