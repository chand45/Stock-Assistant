from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

FUNDAMENTAL_ANALYSIS_SYSTEM_PROMPT = (
    "Perform a fundamental analysis of the stock. Provide a detailed analysis including financial ratios, earnings reports, and other relevant information. Use the ask perplexity tool to search for the fundamental analysis."
)

TECHNICAL_ANALYSIS_SYSTEM_PROMPT = (
    "Perform a technical analysis of the stock. Analyze charts, trends, and technical indicators. Use the ask perplexity tool to search for technical analysis."
)

STOCK_NAME_SYSTEM_PROMPT = (
    "Get the name of the stock to analyze. Only return the name in the format 'Company Name (Ticker)'. Eg: 'Tata Motors Ltd. (NSE:TATAMOTORS)' and nothing else. Only fetch the ticker symbol for NSE. Use the ask perplexity tool to search for the stock ticker."
)

DECISION_SYSTEM_PROMPT = (
    """
    Based on the fundamental analysis: {fundamental_analysis}
    And technical analysis: {technical_analysis}
    
    Make a buy, sell, or hold decision for {stock_name}.
    """
)

llm = init_chat_model(
    "azure_openai:gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

reasoning_llm = init_chat_model(
    "azure_openai:o3",
    azure_deployment=os.environ["AZURE_OPENAI_REASONING_DEPLOYMENT_NAME"],
)

async def get_tools():
    client = MultiServerMCPClient({
        "perplexity": {
            "command": "npx",
            "args": ["chand45-perplexity-ask@latest"],
            "env": {
                "PERPLEXITY_API_KEY": os.environ["PERPLEXITY_API_KEY"]
            },
            "transport": "stdio",
        }
    })
    
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools")
    return tools, client

tools, mcp_client = asyncio.run(get_tools()) # tools = ["perplexity_ask", "perplexity_research", "perplexity_reason"]

llms_with_tools = llm.bind_tools(tools)

class FundamentalAnalysisState(TypedDict):
    stock_name: str
    messages: Annotated[list, add_messages]
    fundamental_analysis: str

async def get_fundamental_analysis(state: FundamentalAnalysisState):
    messages_with_prompt = [
        {"role": "system", "content": FUNDAMENTAL_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"The stock to analyze is {state['stock_name']}"},
        *state['messages']
    ]
    response = await llms_with_tools.ainvoke(messages_with_prompt)

    if not hasattr(response, "tool_calls") or len(response.tool_calls) == 0: # type: ignore
        fundamental_analysis = response.content
        print(f"Fundamental analysis set to: {fundamental_analysis}")
        return {"fundamental_analysis": fundamental_analysis, "messages": [response]}

    return {"messages": [response]}

fsubgraphbuilder = StateGraph(FundamentalAnalysisState)
fsubgraphbuilder.add_node("fundamental_analysis", get_fundamental_analysis)
fsubgraphbuilder.add_node("fundamental_analysis_tools", ToolNode(tools=tools))
fsubgraphbuilder.add_edge(START, "fundamental_analysis")
fsubgraphbuilder.add_conditional_edges(
    "fundamental_analysis",
    tools_condition,
    {
        "tools": "fundamental_analysis_tools",
        "__end__": END
    }
)
fsubgraphbuilder.add_edge("fundamental_analysis_tools", "fundamental_analysis")
fsubgraph = fsubgraphbuilder.compile()


class TechnicalAnalysisState(TypedDict):
    stock_name: str
    messages: Annotated[list, add_messages]
    technical_analysis: str

async def get_technical_analysis(state: TechnicalAnalysisState):
    messages_with_prompt = [
        {"role": "system", "content": TECHNICAL_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"The stock to analyze is {state['stock_name']}"},
        *state['messages']
    ]
    response = await llms_with_tools.ainvoke(messages_with_prompt)

    if not hasattr(response, "tool_calls") or len(response.tool_calls) == 0:
        technical_analysis = response.content
        print(f"Technical analysis set to: {technical_analysis}")
        return {"technical_analysis": technical_analysis, "messages": [response]}

    return {"messages": [response]}

tsubgraphbuilder = StateGraph(TechnicalAnalysisState)
tsubgraphbuilder.add_node("technical_analysis", get_technical_analysis)
tsubgraphbuilder.add_node("technical_analysis_tools", ToolNode(tools=tools))
tsubgraphbuilder.add_edge(START, "technical_analysis")
tsubgraphbuilder.add_conditional_edges(
    "technical_analysis",
    tools_condition,
    {
        "tools": "technical_analysis_tools",
        "__end__": END
    }
)
tsubgraphbuilder.add_edge("technical_analysis_tools", "technical_analysis")
tsubgraph = tsubgraphbuilder.compile()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    stock_name: str
    fundamental_analysis: str
    technical_analysis: str
    decision: Literal["buy", "sell", "hold"]


async def get_stock_name(state: State):
    messages_with_prompt = [
        {"role": "system", "content": STOCK_NAME_SYSTEM_PROMPT},
        *state["messages"]
    ]

    response = await llms_with_tools.ainvoke(messages_with_prompt)

    if not hasattr(response, "tool_calls") or len(response.tool_calls) == 0: # type: ignore
        stock_name = response.content
        print(f"Stock name set to: {stock_name}")
        return {"stock_name": stock_name, "messages": [response]}

    return {"messages": [response]}

async def continue_to_analyses(state: State):
    """Route to both fundamental and technical analysis in parallel"""
    fundamental_task = fsubgraph.ainvoke({"stock_name": state["stock_name"]})
    technical_task = tsubgraph.ainvoke({"stock_name": state["stock_name"]})
    fundamental_result, technical_result = await asyncio.gather(
        fundamental_task, 
        technical_task
    )
    messages = fundamental_result["messages"] + technical_result["messages"]
    return {"fundamental_analysis": fundamental_result["fundamental_analysis"], "technical_analysis": technical_result["technical_analysis"], "messages": messages}

async def make_decision(state: State):
    """Combine both analyses to make a decision"""
    decision_prompt = DECISION_SYSTEM_PROMPT.format(
        fundamental_analysis=state.get('fundamental_analysis', 'Not available'),
        technical_analysis=state.get('technical_analysis', 'Not available'),
        stock_name=state.get('stock_name', 'Unknown')
    )

    response = await reasoning_llm.ainvoke([{"role": "user", "content": decision_prompt}])
    return {"decision": response.content, "messages": [response]}

graph_builder = StateGraph(State)

graph_builder.add_node("stock_name", get_stock_name)
graph_builder.add_node("stock_name_tools", ToolNode(tools=tools))
graph_builder.add_node("continue_to_analyses", continue_to_analyses)
graph_builder.add_node("make_decision", make_decision)

# Edges
graph_builder.add_edge(START, "stock_name")
graph_builder.add_conditional_edges(
    "stock_name",
    tools_condition,
    {
        "tools": "stock_name_tools",
        "__end__": "continue_to_analyses"  # Go to fundamental analysis first
    }
)
graph_builder.add_edge("stock_name_tools", "stock_name")

graph_builder.add_edge("continue_to_analyses", "make_decision")
graph_builder.add_edge("make_decision", END)

memory = InMemorySaver()

stock_graph = graph_builder.compile()