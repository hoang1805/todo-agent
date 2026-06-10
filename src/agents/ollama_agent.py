from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain.agents import create_agent
from typing import List
from langchain_core.tools import BaseTool
from core.llm_client import create_ollama_model
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def create_ollama_agent(model: str, temperature: float, tools: List[BaseTool]):
    """Create a tool-calling agent backed by an Ollama model"""
    llm = create_ollama_model(model, temperature)
    return create_agent(
        llm,
        tools=tools,
        system_prompt="""You are a helpful AI Task Management Assistant. 
Your goal is to help the user organize and manage their daily tasks efficiently. 
When providing tasks to the user, format them clearly and be concise. 
If a task is overdue, politely bring it to their attention.""",
        # middleware=[
        #     SummarizationMiddleware(
        #         model=create_ollama_model(model, temperature, with_thinking=False),
        #         trigger=("tokens", 1500),
        #         keep=("messages", 5)
        #     )
        # ],
        checkpointer=InMemorySaver(),
    )


import streamlit as st

@st.cache_resource
async def get_or_create_agent(model: str, temperature: float):
    tools = [
    ]
    return create_ollama_agent(model, temperature, tools)


def run_task_agent(question: str, model: str, temperature: float, thread_id: str = "1"):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    agent = get_or_create_agent(model, temperature)
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    # result["messages"] is a list; the last message is the final AI response
    response = result["messages"][-1].content
    if not response.strip():
        return "I'm sorry, I encountered an issue and couldn't generate a response. Please try asking again or clearing the cache."
    return response

def clear_agent_thread(model: str, temperature: float, thread_id: str):
    agent = get_or_create_agent(model, temperature)
    if hasattr(agent.checkpointer, "delete_thread"):
        agent.checkpointer.delete_thread(thread_id)

class AgentChunk:
    def __init__(self, type: str, content: str = "", reasoning: str = ""):
        self.type = type
        self.content = content
        self.reasoning = reasoning

def stream_task_agent(question: str, model: str, temperature: float, thread_id: str = "1"):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    agent = get_or_create_agent(model, temperature)
    
    in_think_block = False
    buffer = ""

    for msg, metadata in agent.stream({"messages": [HumanMessage(content=question)]}, config=config, stream_mode="messages"):
        # Ignore chunks from SummarizationMiddleware or other nodes
        if metadata.get("langgraph_node") not in ("agent", "model"):
            continue
            
        if not isinstance(msg, AIMessageChunk):
            continue
            
        # 1. Natively parsed reasoning from langchain_ollama's ChatOllama(reasoning=True)
        # print(msg)
        reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
        if reasoning_content:
            yield AgentChunk(type="reasoning", reasoning=reasoning_content)
            continue
            
        # 2. Raw text processing / Fallback <think> parser for normal models
        if msg.content:
            buffer += msg.content
            
            while True:
                if not in_think_block:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        if idx > 0:
                            yield AgentChunk(type="text", content=buffer[:idx])
                        in_think_block = True
                        buffer = buffer[idx + len("<think>"):]
                    else:
                        split_idx = len(buffer)
                        for i in range(1, min(8, len(buffer) + 1)):
                            if buffer[-i:].startswith("<"):
                                split_idx = len(buffer) - i
                                break
                        if split_idx > 0:
                            yield AgentChunk(type="text", content=buffer[:split_idx])
                            buffer = buffer[split_idx:]
                        break
                else:
                    idx = buffer.find("</think>")
                    if idx != -1:
                        if idx > 0:
                            yield AgentChunk(type="reasoning", reasoning=buffer[:idx])
                        in_think_block = False
                        buffer = buffer[idx + len("</think>"):]
                    else:
                        split_idx = len(buffer)
                        for i in range(1, min(9, len(buffer) + 1)):
                            if buffer[-i:].startswith("<"):
                                split_idx = len(buffer) - i
                                break
                        if split_idx > 0:
                            yield AgentChunk(type="reasoning", reasoning=buffer[:split_idx])
                            buffer = buffer[split_idx:]
                        break
                        
    if buffer:
        if in_think_block:
            yield AgentChunk(type="reasoning", reasoning=buffer)
        else:
            yield AgentChunk(type="text", content=buffer)