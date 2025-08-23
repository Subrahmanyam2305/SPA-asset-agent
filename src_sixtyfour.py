from dataclasses import dataclass
from typing_extensions import override
from dacite import from_dict

import dacite

import requests
import json
import os

from src_base import ResearchSource, ResearchUpdates, Asset

@dataclass
class SixtyFourPayload:
    findings: list[str]
    latest_news: str
    latest_industry_news: str
    num_employees: str
    num_employees_us: str
    last_executive_exits: str
    last_executive_joins: str
    glassdoor_chatter: str
    NGMI: str

@dataclass
class SixtyFourResearchUpdates(ResearchUpdates):
    details: SixtyFourPayload

class SixtyFourResearchSource(ResearchSource):
    @override
    def research_asset_update(self, asset: Asset) -> SixtyFourResearchUpdates:
        response = requests.post(
            "https://api.sixtyfour.ai/enrich-company",
            headers={
                "x-api-key": os.getenv("SIXTYFOUR_API_KEY"),
                "Content-Type": "application/json",
            },
            json={
                "target_company": {
                    "company_name": "Digital Realty",
                    "website": "https://www.digitalrealty.com"
                },
                "struct": {
                    "latest_news": "The news about the company for the last 3 months, long form, most relevant first",
                    "latest_industry_news": "The news about company's industry for the last 1 month, long form",
                    "num_employees": "How many employees work there, give approximate estimation if no number available",
                    "num_employees_us": "How many employees the company has working in the US",
                    "last_executive_exits": "What are the executives that exited the company in the last 3 months",
                    "last_executive_joins": "What are the new executives that joined the company in the last 3 months",
                    "glassdoor_chatter": "Find and aggregate latest (<3 months old) employee reviews on Glassdoor, aggregate sentiment types and prominence of each type, provide typical quote examples",
                    "NGMI": """Does the company have risk of becoming disstressed, provide risk estimation in % and reasoning"""
                },
                "find_people": True,
                "research_plan": "Check latest news, Glassdoor employee reviews, latest employee updates, etc",
                "people_focus_prompt": "Find me the key current and former (left <3 months back) executives in the company, collect information about their current job, previous job, how long they worked at the company"
            },
        )

        data = json.loads(response.content)
        sfp = from_dict( SixtyFourPayload,
            data = (data["structured_data"] | { "findings": data["findings"] }),
            config = dacite.Config(strict=False)
        )

        return SixtyFourResearchUpdates(sfp.NGMI, sfp)

    @override
    def initialized() -> bool:
        raise True #TODO
    
    @override
    def initialize_asset(self, asset: Asset) -> None:
        pass