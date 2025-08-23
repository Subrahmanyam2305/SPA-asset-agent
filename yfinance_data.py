import yfinance as yf
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class YFinance:
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

    def get_research_updates(self, company_name: str) -> Optional[str]:
        """
        Get research updates for a company using yfinance.
        
        Args:
            company_name (str): Name of the company
            
        Returns:
            Optional[str]: JSON string containing research data if available,
                         None if company doesn't have a ticker
        """
        ticker_symbol = self._get_ticker(company_name)
        if not ticker_symbol:
            return None

        try:
            # Get ticker information
            ticker = yf.Ticker(ticker_symbol)
            
            # Collect relevant information
            research_data = {
                "company_name": company_name,
                "ticker": ticker_symbol,
                "info": {},
                "news": [],
                "analyst_recommendations": [],
                "timestamp": datetime.now().isoformat()
            }

            # Basic company information
            try:
                info = ticker.info
                research_data["info"] = {
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
                research_data["info"] = {"error": f"Failed to fetch company info: {str(e)}"}

            # Recent news
            try:
                news = ticker.news
                if news:
                    research_data["news"] = [
                        {
                            "title": item.get("title"),
                            "publisher": item.get("publisher"),
                            "link": item.get("link"),
                            "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat()
                        }
                        for item in news[:5]  # Get latest 5 news items
                    ]
            except Exception as e:
                research_data["news"] = [{"error": f"Failed to fetch news: {str(e)}"}]

            # Analyst recommendations
            try:
                recommendations = ticker.recommendations
                if not recommendations.empty:
                    recent_recommendations = recommendations.tail(5)  # Get latest 5 recommendations
                    research_data["analyst_recommendations"] = [
                        {
                            "firm": row.get("Firm"),
                            "to_grade": row.get("To Grade"),
                            "action": row.get("Action"),
                            "date": row.name.isoformat()
                        }
                        for _, row in recent_recommendations.iterrows()
                    ]
            except Exception as e:
                research_data["analyst_recommendations"] = [{"error": f"Failed to fetch recommendations: {str(e)}"}]

            return json.dumps(research_data, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Failed to fetch data for {company_name} ({ticker_symbol}): {str(e)}",
                "timestamp": datetime.now().isoformat()
            })


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
