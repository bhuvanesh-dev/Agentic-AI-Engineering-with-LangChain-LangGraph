from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field


class SourcesModel(BaseModel):
    """
    schema for agent sources. It contains the url of the source for the answer.
    """
    url: str = Field(description="The agent's source url for the answer.")

class ResultModel(BaseModel):
    """
    Schema for agent response. It contains the agent's answer and the sources for the answer.
    """
    answers: str = Field(description="The agent's answer to the question.")
    sources: list[SourcesModel] = Field(default_factory=list, description="The agent's sources for the answer.")


tavlily_searchClient = TavilyClient()

@tool
def search(query: str) -> str:
    """
    This tool performs a search for the given query in internet and returns the results as a string.
    Args:
        query (str): The search query to be performed.
    Returns:
        str: Search results for query
    """
    return tavlily_searchClient.search(query = query)

def main():
    print("Hello from langchain-course!")
    agent = create_agent(
        tools=[search],
        model=ChatOllama(model="qwen3:latest"),
        response_format=ResultModel
    )
    response = agent.invoke({"messages":[HumanMessage("What is the weather in Chennai?")]})
    print(response["structured_response"])


if __name__ == "__main__":
    main()
