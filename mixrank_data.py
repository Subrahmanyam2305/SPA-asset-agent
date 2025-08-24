import requests
import os
from dotenv import load_dotenv

from src_base import Asset, ResearchSource, ResearchUpdates

load_dotenv()

class MixRank(ResearchSource):
    def __init__(self):
        self.api_key = os.getenv("MIXRANK_API_KEY")
        # api key is in the .env file but it is not being read in
        self.base_url = f"https://api.mixrank.com/v2/json/{self.api_key}"

    def research_asset_update(self, asset: Asset) -> ResearchUpdates | None:
        updates = self.get_research_updates(asset.name)

        if updates == None: return None
        else:
            return ResearchUpdates(relevance_reasoning=str(updates))

    def get_research_updates(self, company_name: str) -> str | None:
        mixrank_content = ""
        endpoint = f"/companies?search={company_name}&page_size=1"
        url = self.base_url + endpoint
        # print(url)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # print(data)
            id = data['results'][0]['id']
            mixrank_content += f"Company name: {data['results'][0]['name']}\n"
            mixrank_content += f"Number of current employees at this company: {data['results'][0]['employees']}\n"
            mixrank_content += f"Inc. 5000 Rank: {data['results'][0]['rank_incmagazine']}\n"
            mixrank_content += f"Fortune 500 Rank: {data['results'][0]['rank_fortune']}\n"

            timeseries_endpoint = f"/companies/{id}/timeseries?page_size=10"
            timeseries_url = self.base_url + timeseries_endpoint
            timeseries_response = requests.get(timeseries_url)
            # return employee count timeseries in a list of tuples (date, employee count)
            if timeseries_response.status_code == 200:
                timeseries_data = timeseries_response.json()
                employee_count_timeseries = [(i['date'], i['employee_count']) for i in timeseries_data['results']]
                mixrank_content += f"Employee count timeseries: {employee_count_timeseries}\n"
            else:
                mixrank_content += f"Error getting timeseries data: {timeseries_response.status_code}\n"         
        else:
            print(f"Error getting  data: {response.status_code}")
            return None
        return mixrank_content
