import os
import json
import re
import textwrap
import dacite
import requests
from dataclasses import dataclass, InitVar
from enum import Enum

from src_base import Asset

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
import ast

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages] # type: ignore
    lastAgent: str

load_dotenv()

llm = init_chat_model("openai:gpt-5-nano")
llm_mini = init_chat_model("openai:gpt-5-nano")

def stream_graph_updates(graph: CompiledStateGraph[Any, Any], user_input: str):
    last_event = None
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            last_event = value["messages"][-1].content
            print("\nAssistant:", last_event)
    return last_event

def withSystemMessage(state: MessagesState, msg: SystemMessage) -> MessagesState:
    messages = state["messages"]
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        messages.insert(0, msg)
    else:
        messages[0] = msg
    return state

def addToolEdge(graph_builder: StateGraph[Any, Any, Any, Any], source: str, tools: str, destination: str) -> StateGraph[Any, Any, Any, Any]:
    return graph_builder.add_conditional_edges(
        source,
        tools_condition,
        {"tools": tools, END: destination},
    )

tool_search = TavilySearch(max_results=10)
tool_extract = TavilyExtract()

tools_research = [tool_search]
tools_deep_research = [tool_search, tool_extract]

# Modification: tell the LLM which tools it can call
llm_research = llm_mini.bind_tools(tools_research)
llm_deep_research = llm_mini.bind_tools(tools_deep_research)

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    ABSENT = "absent"

@dataclass
class RiskAssessorOutput:
    risk_level: RiskLevel
    confidence: int
    reasoning: str

llm_risk_assessor = llm_deep_research.with_structured_output(RiskAssessorOutput)

def risk_assessor(asset: Asset, state: State):
    print("\nAgent: Risk Assessor\n")

    risk_assessor_system_message: SystemMessage = SystemMessage(
    textwrap.dedent("""You are an investment system's asset early warning risk assessor. Your job is to analyze incoming data to detect weak signals that may require special attention.
        You only observe and detect signals that can affect the asset and assess each signel risk level implications separately.

        Rules:
        - If you see potential for risk, but you don't have enough data, call search and extract to get more information
        - Call at most THREE tools per turn
        - Do not call the same tool with the same input twice
        - Stop when you believe you have enough signal or after 5 tool calls max
        """))

    stateDelta = {"messages": []}

    stateDelta["lastAgent"] = "risk_assessor"
    stateDelta["messages"].append(
        AIMessage(json.dumps(llm_risk_assessor.invoke(
            [risk_assessor_system_message] + state["messages"]
        )))
    )

    return stateDelta

@dataclass
class ReporterOutput:
    risk_level: RiskLevel
    report: str
    sources: list[str]

llm_reporter = llm.with_structured_output(ReporterOutput)

def reporter(state: State):
    print("\nAgent: Reporter\n")

    risk_reporter_system_message: SystemMessage = SystemMessage(
    textwrap.dedent("""You are a report generating agent. Your task is to synthesize insights collected by risk assessor into a detailed report in MD format.
        Your output is a report on how a given event provided to you affects the risk level also provided to you.

        Rules:
        - Format your output in MD in report field, cite sources and put them to sources list
        """))

    stateDelta = {"messages": []}

    stateDelta["lastAgent"] = "reporter"
    stateDelta["messages"].append(
        AIMessage(json.dumps(llm_reporter.invoke(
            [risk_reporter_system_message] + state["messages"]
        )))
    )

    return stateDelta

pipelines: dict[str, CompiledStateGraph] = {}

def build_pipeline(asset: Asset):
    if asset.name in pipelines:
        return pipelines[asset.name]

    graph_builder = StateGraph(State)

    graph_builder.add_node("risk-assessor", partial(risk_assessor, "Digital Realty"))
    graph_builder.add_node("reporter", reporter)

    deep_research_tool_node = ToolNode(tools=[tool_search, tool_extract])
    graph_builder.add_node("tools-risk-assessor", deep_research_tool_node)

    graph_builder.add_edge(START, "risk-assessor")

    addToolEdge(graph_builder, "risk-assessor", "tools-risk-assessor", "reporter")
    graph_builder.add_edge("tools-risk-assessor", "risk-assessor")

    graph_builder.add_edge("reporter", END)

    graph = graph_builder.compile()

    return graph