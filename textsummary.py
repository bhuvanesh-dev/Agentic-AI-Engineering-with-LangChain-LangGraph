from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def main():
    print("Hello from langchain-course!")
    infromation = """Chempakaraman Pillai (alias Venkidi;[1] 15 September 1891 – 26 May 1934) was an Indian-born political activist and revolutionary.[2] Born in Thiruvananthapuram, he left for Europe as a youth, where he spent the rest of his active life as an Indian nationalist and revolutionary.[3]

Although his life was mired in controversies, including a squabble with Adolf Hitler,[4] information on his life in Europe was sketchy in the immediate years after his death. More information has come out in recent years.[1]

Chempakaraman Pillai is credited with the coining of the salutation and slogan "Jai Hind"[1][5] in the pre-independence days of India. The slogan is still widely used in India.

Pillai, who started the Indian National Voluntary Corps on 31 July 1914, was instrumental in inspiring Netaji Subhas Chandra Bose to start the Indian National Army (INA).[4]"""
    summary_template = """      You are a helpful assistant that summarizes information about a person. information: {information}
              1. Provide a concise summary.
              2. Provide two interesting facts about the person."""
    summary_template_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    llm = ChatOllama(temperature=0, model="qwen3:latest")
    chain = summary_template_prompt | llm
    response = chain.invoke(input={"information": infromation})
    print(response.content)
#sent enviromnet variable for apikey in terminal export OLLAMA_API_KEY
os.


if __name__ == "__main__":
    main()