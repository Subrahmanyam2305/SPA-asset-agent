import yfinance
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from src_base import ResearchSource, ResearchUpdates, Asset

@dataclass
class YFinanceResearchUpdates(ResearchUpdates):
    company_info: dict
    news: list
    analyst_recommendations: list
    ticker: str

class YFinance(ResearchSource):
    def __init__(self):
        """Initialize the YFinance class with company to ticker mapping."""
        self.company_tickers = {
            "google": "GOOGL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "facebook": "META",  # Updated ticker for Facebook/Meta
            "apple": "AAPL",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "amd": "AMD",
            "intel": "INTC"
            # Note: OpenAI is not publicly traded, so no ticker
        }

    def _get_ticker(self, company_name: str) -> Optional[str]:
        """
        Get the stock ticker for a company name.
        
        Args:
            company_name (str): Name of the company
            
        Returns:
            Optional[str]: Stock ticker if available, None if not found
        """
        return self.company_tickers.get(company_name.lower())

    def name(self) -> str:
        return "YFinance"

    def initialized(self) -> bool:
        return True  # No initialization needed

    def initialize_asset(self, asset: Asset) -> None:
        # No initialization needed
        pass

    def research_asset_update(self, asset: Asset) -> YFinanceResearchUpdates | None:
        """
        Get research updates for a company using yfinance.
        
        Args:
            asset (Asset): Asset to research
            
        Returns:
            YFinanceResearchUpdates | None: Research updates if available,
                                          None if company doesn't have a ticker
        """
        ticker_symbol = self._get_ticker(asset.name)
        if not ticker_symbol:
            return None

        try:
            # Get ticker information
            ticker = yfinance.Ticker(ticker_symbol)
            
            # Basic company information
            try:
                info = ticker.info
                company_info = {
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "forward_pe": info.get("forwardPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "website": info.get("website"),
                    "business_summary": info.get("longBusinessSummary")
                }
            except Exception as e:
                company_info = {"error": f"Failed to fetch company info: {str(e)}"}

            # Recent news
            news_items = []
            try:
                news = ticker.news
                if news:
                    news_items = [
                        {
                            "title": item['content'].get("title"),
                            "summary": item['content'].get("summary"),
                            "description": item['content'].get("description"),
                            "published": item['content'].get("pubDate")
                        }
                        for item in news[:5]  # Get latest 5 news items
                    ]
            except Exception as e:
                news_items = [{"error": f"Failed to fetch news: {str(e)}"}]

            # Analyst recommendations
            analyst_recs = []
            try:
                recommendations = ticker.recommendations
                if not recommendations.empty:
                    recent_recommendations = recommendations.tail(5)  # Get latest 5 recommendations
                    analyst_recs = [
                        {
                            "period": row.get("period"),
                            "strongBuy": row.get("strongBuy"),
                            "buy": row.get("buy"),
                            "hold": row.get("hold"),
                            "sell": row.get("sell"),
                            "strongSell": row.get("strongSell")
                        }
                        for _, row in recent_recommendations.iterrows()
                    ]
            except Exception as e:
                analyst_recs = [{"error": f"Failed to fetch recommendations: {str(e)}"}]

            # Create relevance reasoning
            relevance = (
                f"Financial market data for {asset.name} ({ticker_symbol}). "
                f"Includes company information, recent news, and analyst recommendations. "
                f"Market cap: {company_info.get('market_cap', 'N/A')}, "
                f"Industry: {company_info.get('industry', 'N/A')}"
            )

            return YFinanceResearchUpdates(
                relevance_reasoning=relevance,
                company_info=company_info,
                news=news_items,
                analyst_recommendations=analyst_recs,
                ticker=ticker_symbol
            )

        except Exception as e:
            # Return None on error to indicate no data available
            return None


if __name__ == "__main__":
    # Example usage
    yf = YFinance()
    companies = ["google", "openai", "microsoft"]  # openai will be skipped
    for company in companies:
        result = yf.get_research_updates(company)
        if result:
            print(f"\n=== {company.upper()} ===")
            print(result)
        else:
            print(f"\nNo ticker found for {company}")