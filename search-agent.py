from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """
    This tool performs a search for the given query in internet and returns the results as a string.
    Args:
        query (str): The search query to be performed.
    Returns:
        str: Search results for query
    """
    return "Chennai weather is 30 degree celsius"

def main():
    print("This is an search agent!")
    tools = [search]
    llm = ChatOllama(model="qwen3:latest", temperature=0.9)
    agent = create_agent(tools=tools, model=llm)
    response = agent.invoke({"messages":HumanMessage(content="What is the weather in Chennai?")})
    print(response)

if __name__ == "__main__":
    main()