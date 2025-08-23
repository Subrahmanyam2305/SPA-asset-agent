import os
import json
from queue import Queue
import re
import textwrap
import dacite
import requests
from dataclasses import dataclass, InitVar, asdict
from enum import Enum
import concurrent.futures
import traceback

from src_base import Asset, ResearchSource

from dotenv import load_dotenv

from pprint import pprint

from functools import partial

from typing import Annotated, Any
from typing_extensions import TypedDict

from IPython.display import Image, display
from requests import Response

from pydantic import BaseModel, HttpUrl

from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send
from langgraph.graph.state import StateGraph, CompiledStateGraph

from langgraph.checkpoint.memory import InMemorySaver

from langchain_tavily import TavilySearch, TavilyExtract

import src_sixtyfour as sixtyfour
import mixrank_data as mixrank
import ast

import sqlite3

import agentic_pipeline

load_dotenv()

def load_assets(db_name: str) -> dict[str, Asset]:
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT asset_name
                FROM research_docs
            """)
            
            assets = []
            rows = cursor.fetchall()
            for row in rows:
                asset_name = row[0]
                assets.append(Asset(asset_name, None))
            return assets
    except sqlite3.Error as e:
        print(f"Error initializing asset {asset_name}: {e}")
        raise

def load_sources() -> list[ResearchSource]:
    return [
        sixtyfour.SixtyFourResearchSource(),
        mixrank.MixRank(),
    ]

def get_thread_pool(n: int):
    return concurrent.futures.ThreadPoolExecutor(max_workers=n)

def process_asset(asset: Asset, sources: list[ResearchSource]):
    graph = agentic_pipeline.build_pipeline(asset)
    sixty_four = sixtyfour.SixtyFourResearchSource()

    for source in sources:
        try:
            print(f"Processing source: {source.name}")

            insight = source.research_asset_update(asset)

            user_input = (
                f"Please analyze the following data set from {source.name} to assess for potential risks for the asset {asset.name}.\n\n"
                f"{(json.dumps(asdict(insight)))}"
            )
            
            result_str: str = agentic_pipeline.stream_graph_updates(graph, user_input)
            result = dacite.from_dict(agentic_pipeline.ReporterOutput, json.loads(result_str),
                dacite.Config(strict=False, type_hooks={
                    agentic_pipeline.RiskLevel: lambda v: v if isinstance(v, agentic_pipeline.RiskLevel) else agentic_pipeline.RiskLevel(v)
                }))
            print("\n\n\n\n")
            pprint(result)
        except Exception as e:
            traceback.print_exc()
    
if __name__ == "__main__":
    assets = load_assets("research_docs.db")
    sources = load_sources()

    assets = [Asset("Digital Realty", None)]
    
    assets_queue: Queue[Asset] = Queue()
    for a in assets: assets_queue.put(a)

    pool = get_thread_pool(5)
    futures = []

    for a in assets:
        futures.append(pool.submit(lambda: process_asset(a, sources)))
    
    concurrent.futures.wait(futures)