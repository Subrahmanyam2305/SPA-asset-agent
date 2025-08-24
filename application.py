import os
import json
from queue import Empty, Queue
import re
import textwrap
from uuid import UUID
import dacite
import requests
from dataclasses import dataclass, InitVar, asdict
from enum import Enum
import concurrent.futures
import traceback

import init
from src_base import Asset, ResearchSource, ResearchUpdates

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

def process_asset(asset: Asset, sources: list[ResearchSource], results_out: Queue[(Asset, list[ResearchUpdates])]) -> list[agentic_pipeline.ReporterOutput]:
    try:
        graph = agentic_pipeline.build_pipeline(asset)

        results = []
        for source in sources:
            try:
                print(f"Processing source: {source.name()} for asset {asset.name}")

                insight = source.research_asset_update(asset)

                user_input = (
                    f"Please analyze the following data set from {source.name} to assess for potential risks for the asset {asset.name}.\n\n"
                    f"{(json.dumps(asdict(insight)))}"
                )
                
                result_str: str = agentic_pipeline.stream_graph_updates(graph, user_input, config={"configurable": {"thread_id": asset.name}}, output=False)

                # data = json.loads(result_str)
                # if isinstance(data.get("sources"), str):
                #     import re
                #     data["sources"] = [s.strip() for s in re.split(r"[,\n]+", data["sources"]) if s.strip()]

                # result = dacite.from_dict(agentic_pipeline.ReporterOutput, data,
                #     dacite.Config(strict=False, type_hooks={
                #         agentic_pipeline.RiskLevel: lambda v: v if isinstance(v, agentic_pipeline.RiskLevel) else agentic_pipeline.RiskLevel(v)
                #     }))
                #results.append(result)
            except Exception as e:
                traceback.print_exc()
    
        data = graph.invoke({"do_report": True}, {"configurable": {"thread_id": asset.name}})

        if isinstance(data.get("sources"), str):
            import re
            data["sources"] = [s.strip() for s in re.split(r"[,\n]+", data["sources"]) if s.strip()]

        result = dacite.from_dict(agentic_pipeline.ReporterOutput, json.loads(data["messages"][-1].content),
                    dacite.Config(strict=False, type_hooks={
                        agentic_pipeline.RiskLevel: lambda v: v if isinstance(v, agentic_pipeline.RiskLevel) else agentic_pipeline.RiskLevel(v)
                    }))

        results.append(result)
    except Exception as e:
        traceback.print_exc()

    results_out.put((asset, results))

def save_report(db_name: str, asset: Asset, report: agentic_pipeline.ReporterOutput):
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO research_reports (asset_name, report)
                VALUES (?, ?)
                ON CONFLICT(asset_name) DO UPDATE SET report=excluded.report
            """, (asset.name, report.report))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving report for {asset.name}: {e}")
        raise

worker_pool = get_thread_pool(5)

def run_update(db_name: str, assets = None, sources = None):
    if assets == None: assets = load_assets(db_name)
    if sources == None: sources = load_sources()

    # assets = assets[0:1]
    # sourceds = sources[0:1]

    results: Queue[(Asset, list[ResearchUpdates])] = Queue()
    futures = []

    for a in assets:
        futures.append(worker_pool.submit(lambda: process_asset(a, sources, results)))
    
    concurrent.futures.wait(futures)
    try:
        while (it := results.get_nowait()) is not None:
            pprint(it)
    except Empty:
        pass
    
if __name__ == "__main__":
    init.init_db("research_docs.db")
    run_update("research_docs.db")