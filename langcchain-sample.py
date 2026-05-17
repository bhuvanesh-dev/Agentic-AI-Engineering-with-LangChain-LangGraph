from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


infromationOnElonMusk = """Elon Musk is a business magnate and investor. He is the founder, 
CEO and chief engineer of SpaceX; angel investor, CEO and product architect of Tesla, Inc.;
 owner and CEO of Twitter, Inc.; owner and CEO of xAI;
 president of the philanthropic Musk Foundation;
 and president of the philanthropic Future of Life Institute.
 He was also co-founder and CEO of PayPal and co-founder of Neuralink and OpenAI, among other roles. """

prompt_template = """You are a helpful assistant that provides information about Elon Musk. Information: {infromationOnElonMusk}
    1. Give a oneline summary of the person.
    2. Provide two interesting facts about the person.
"""
prompt = PromptTemplate(input_variables=["infromationOnElonMusk"], template=prompt_template)
llm = ChatOllama(model="qwen3:latest", temperature=0.9)
chain = prompt | llm
response = chain.invoke({"infromationOnElonMusk": infromationOnElonMusk})
print(response.content)